#!/usr/bin/env python3
"""Build the v5 domain-balanced German RAG training mix (stdlib, no ML, no network).

Reads heterogeneous v5 input JSONL, enforces the v5 hard gates (known license, no public-
benchmark/eval leakage, FAQ share <= --max-faq-share, non-FAQ share >= --min-nonfaq-share),
deterministically domain-balances toward --target-count, and writes a coverage report proving
non-FAQ coverage BEFORE any teacher scoring. Non-zero exit on any hard failure (also in
--dry-run). `--dry-run` writes the report but not the pairs file and imports no torch.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import v5_data_mixer as M  # noqa: E402
from boldt_embed.v5_rag_config import load_v5_rag_config  # noqa: E402


def _read_jsonl(path: pathlib.Path) -> list:
    # split("\n"), not splitlines(): WebFAQ/web text carries U+2028/U+2029 which
    # str.splitlines() over-splits (json tolerates them inside strings).
    rows = []
    for ln in path.read_text(encoding="utf-8").split("\n"):
        if ln.strip():
            rows.append(json.loads(ln))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="v5 small-RAG config (validated)")
    ap.add_argument("--inputs", nargs="+", required=True, help="input JSONL files (globs pre-expanded)")
    ap.add_argument("--output", required=True, help="output mixed-pairs JSONL")
    ap.add_argument("--report", required=True, help="coverage report JSON")
    ap.add_argument("--target-count", type=int, default=100000)
    ap.add_argument("--max-faq-share", type=float, default=0.35)
    ap.add_argument("--min-nonfaq-share", type=float, default=0.50)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Validates the config (legal diagnostic-only, public-benchmarks-eval-only, etc.).
    cfg = load_v5_rag_config(args.config)

    rows = []
    missing = []
    for p in args.inputs:
        path = pathlib.Path(p)
        if not path.exists():
            missing.append(p)
            continue
        rows.extend(_read_jsonl(path))
    if missing:
        print(f"ERROR: input file(s) not found: {missing}", file=sys.stderr)
        return 2
    if not rows:
        print("ERROR: no input rows read", file=sys.stderr)
        return 2

    # Cross-check: every input domain must be a configured train domain.
    cfg_domains = set(cfg.train_domains)
    stray = sorted({r.get("domain") for r in rows
                    if isinstance(r, dict) and r.get("domain") not in cfg_domains})
    report = M.mix(rows, target_count=args.target_count,
                   max_faq_share=args.max_faq_share, min_nonfaq_share=args.min_nonfaq_share)
    if stray:
        report["status"] = "fail"
        report["errors"].append(f"input domains not in config train_domains: {stray}")

    if args.dry_run:
        assert "torch" not in sys.modules, "dry-run must not import torch"

    selected = report.pop("selected")
    out_report = pathlib.Path(args.report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[v5-mix] status={report['status']} selected={report['selected_rows']} "
          f"faq={report['faq_share']} nonfaq={report['nonfaq_share']} "
          f"synthetic={report['synthetic_share']} -> report {out_report}")
    print(f"[v5-mix] domains: {json.dumps(report['rows_by_domain'], ensure_ascii=False)}")
    for e in report["errors"]:
        print(f"  ✗ {e}", file=sys.stderr)

    if report["status"] != "pass":
        print("FAIL — v5 data mixture rejected (see report)", file=sys.stderr)
        return 1

    if args.dry_run:
        print("dry-run-ok (no ML imports; report written, pairs file NOT written)")
        return 0

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[v5-mix] wrote {len(selected)} pairs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
