#!/usr/bin/env python3
"""Calibrate teacher thresholds and split the positive set per consumer (stdlib, no ML).

The embedder and the reranker should NOT consume the same noisy positive set (v2's mistake).
This reports threshold sensitivity (acceptance at -2/0/1/2/3/4/5, overall + by
domain/source/license), then writes two filtered training sets: the embedder set at the looser
threshold (default 2.0) and a higher-precision reranker set at a stricter threshold (default 4.0),
with optional per-domain overrides from the v3 config. Gates: zero unknown-license rows,
real-domain accepted floors, suspicious-positive-rate cap. Exit non-zero blocks training.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import teacher_calibration as tc  # noqa: E402


def _read_jsonl(path):
    return [json.loads(l) for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _load_cache(path):
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


def _write_jsonl(path, rows):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-cache", required=True, help="cache JSONL / *.manifest.json / glob")
    ap.add_argument("--config", default=None, help="v3 config (teacher_calibration + gates)")
    ap.add_argument("--embedder-threshold", type=float, default=None)
    ap.add_argument("--reranker-threshold", type=float, default=None)
    ap.add_argument("--max-suspicious-rate", type=float, default=None)
    ap.add_argument("--output", required=True, help="calibration JSON")
    ap.add_argument("--markdown", required=True, help="calibration markdown")
    ap.add_argument("--embedder-output", default=None, help="filtered embedder positives JSONL")
    ap.add_argument("--reranker-output", default=None, help="filtered reranker positives JSONL")
    args = ap.parse_args()

    rows = _load_cache(args.teacher_cache)

    cal_cfg, gate_cfg = {}, {}
    if args.config and pathlib.Path(args.config).exists():
        cfg = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
        cal_cfg = cfg.get("teacher_calibration") or {}
        gate_cfg = cfg.get("domain_quality_gates") or {}

    emb_t = args.embedder_threshold if args.embedder_threshold is not None \
        else float(cal_cfg.get("embedder_threshold", tc.DEFAULT_EMBEDDER_THRESHOLD))
    rr_t = args.reranker_threshold if args.reranker_threshold is not None \
        else float(cal_cfg.get("reranker_threshold", tc.DEFAULT_RERANKER_THRESHOLD))
    max_susp = args.max_suspicious_rate if args.max_suspicious_rate is not None \
        else float(cal_cfg.get("max_suspicious_positive_rate", tc.DEFAULT_MAX_SUSPICIOUS_RATE))

    report = tc.calibrate(
        rows, embedder_threshold=emb_t, reranker_threshold=rr_t,
        per_domain_embedder=cal_cfg.get("per_domain_embedder"),
        per_domain_reranker=cal_cfg.get("per_domain_reranker"),
        min_real_domain_accepted=gate_cfg.get("min_real_domain_accepted"),
        max_suspicious_rate=max_susp)

    emb_kept = report.pop("_embedder_kept")
    rr_kept = report.pop("_reranker_kept")
    if args.embedder_output:
        _write_jsonl(args.embedder_output, emb_kept)
    if args.reranker_output:
        _write_jsonl(args.reranker_output, rr_kept)

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pathlib.Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.markdown).write_text(tc.render_markdown(report), encoding="utf-8")

    print(json.dumps({"status": report["status"], "embedder_accepted": report["embedder_accepted"],
                      "reranker_accepted": report["reranker_accepted"],
                      "suspicious_positive_rate": report["suspicious_positive_rate"],
                      "failing_gates": report["failing_gates"]}, ensure_ascii=False, indent=2))
    print(f"[calibrate] status={report['status']} embedder={report['embedder_accepted']} "
          f"reranker={report['reranker_accepted']} -> {args.output}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
