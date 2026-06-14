#!/usr/bin/env python3
"""Analyze v3 domain quality and enforce gates BEFORE training (pure stdlib, no ML/network).

Compares the raw candidate set against the teacher cache and reports raw/accepted by
domain/source/license, acceptance rates, effective distribution, synthetic-vs-real share, score
medians, suspicious positives, and license/provenance counts — then evaluates the v3 gates.
Exit non-zero (so training is blocked) when any gate fails.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import domain_quality as dq  # noqa: E402

DEFAULT_OUT = ROOT / "outputs" / "v3-real-domain" / "domain_quality.json"
DEFAULT_MD = ROOT / "outputs" / "v3-real-domain" / "domain_quality.md"


def _read_jsonl(path):
    return [json.loads(l) for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _load_cache(path):
    """Teacher cache: a JSONL, a *.manifest.json (read its shards), or a glob."""
    p = pathlib.Path(path)
    if p.name.endswith(".manifest.json"):
        man = json.loads(p.read_text(encoding="utf-8"))
        rows = []
        for sh in man.get("shards", []):
            sp = pathlib.Path(sh["path"])
            if sp.exists():
                rows += _read_jsonl(sp)
        return rows
    if any(ch in str(path) for ch in "*?["):
        rows = []
        for f in sorted(glob.glob(str(path))):
            rows += _read_jsonl(f)
        return rows
    return _read_jsonl(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", required=True, help="candidate JSONL")
    ap.add_argument("--teacher-cache", required=True, help="teacher cache JSONL / *.manifest.json / glob")
    ap.add_argument("--config", default=None, help="v3 config JSON (may carry domain_quality_gates)")
    ap.add_argument("--source-manifest", default=None,
                    help="optional v3 source manifest (to flag supplemental + disallowed sources)")
    ap.add_argument("--reranker-threshold", type=float, default=2.0)
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--markdown", default=str(DEFAULT_MD))
    args = ap.parse_args()

    candidates = _read_jsonl(args.candidates)
    cache_rows = _load_cache(args.teacher_cache)

    gates = None
    if args.config and pathlib.Path(args.config).exists():
        gates = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8")).get(
            "domain_quality_gates")

    supplemental, disallowed = set(), set()
    if args.source_manifest and pathlib.Path(args.source_manifest).exists():
        from boldt_embed import domain_source_acquisition as dsa
        entries = dsa.load_v3_manifest(args.source_manifest)
        supplemental = {e.source_id for e in entries if e.supplemental}
        disallowed = {e.source_id for e in entries if not e.allowed_for_training}

    report = dq.analyze(candidates, cache_rows, gates=gates,
                        reranker_threshold=args.reranker_threshold,
                        supplemental_sources=supplemental, disallowed_sources=disallowed)

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    pathlib.Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.markdown).write_text(dq.render_markdown(report), encoding="utf-8")

    print(json.dumps({k: report[k] for k in ("status", "totals", "failing_gates",
                                             "can_claim_legal_transfer_from_data")},
                     ensure_ascii=False, indent=2))
    print(f"[domain-quality] status={report['status']} "
          f"failing_gates={len(report['failing_gates'])} -> {args.output}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
