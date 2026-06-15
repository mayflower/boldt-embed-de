#!/usr/bin/env python3
"""Analyze the catastrophic GermanQuAD reranking drops (stdlib, no ML, no training). For each query
where reranking catastrophically drops nDCG@10 vs the first stage, classify the error type and check
which bounded policy would fix it — telling us whether the remaining failures are POLICY-fixable or
DATA/MODEL-fixable. Writes a JSON report + a Markdown summary (counts by error type, fixability,
top-20 examples).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import rerank_error_analysis as EA  # noqa: E402


def _read(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", default=None, help="policy eval json (optional; for scoping/labeling)")
    ap.add_argument("--lists", required=True, help="scored GermanQuAD candidate lists JSONL")
    ap.add_argument("--output", required=True)
    ap.add_argument("--markdown", required=True)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    lp = pathlib.Path(args.lists)
    if not lp.exists():
        print(f"ERROR: candidate lists not found: {lp}", file=sys.stderr)
        return 2
    rows = _read(lp)
    report = EA.analyze(rows)

    if args.eval:
        ev = pathlib.Path(args.eval)
        if ev.exists():
            e = json.loads(ev.read_text(encoding="utf-8"))
            report["policy_eval_context"] = {
                "policy": e.get("policy_name") or e.get("policy"),
                "policy_catastrophic_drop_rate": e.get("catastrophic_drop_rate"),
                "policy_delta_vs_first_stage": e.get("delta_vs_first_stage")}

    assert "torch" not in sys.modules, "analysis must not import torch"
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    pathlib.Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.markdown).write_text(EA.render_markdown(report, top=args.top), encoding="utf-8")

    print(f"[catastrophic-analysis] {report['n_catastrophic']}/{report['n_queries_total']} "
          f"catastrophic; policy-fixable {report['fixable_by_any_policy']} "
          f"({report['policy_fixable_fraction']*100:.1f}%), data/model-fixable "
          f"{report['not_fixable_by_any_policy']} -> {args.output}")
    print(f"[catastrophic-analysis] error types: {report['counts_by_error_type']}")
    print(f"[catastrophic-analysis] fixed_by: {report['fixable_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
