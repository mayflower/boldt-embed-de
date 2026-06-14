#!/usr/bin/env python3
"""v4 RAG reranker promotion gate (pure stdlib). Reads the reranker_lift_*.json reports in
--eval-dir and PASSES only if the reranker is RAG-useful AND does not degrade public benchmarks:

  WebFAQ held-out delta_ndcg@10 >= 0.03, local_rag >= 0.03 (if present),
  GermanQuAD & DT-test delta >= 0.0, no eval set drops more than -0.02,
  first-stage recall high enough for reranking to matter, all sets fixed candidate lists.

GerDaLIR / legal are DIAGNOSTIC ONLY — reported, never a gate. Exit 0 = pass, 1 = fail.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import rag_reranker_eval as RE  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", required=True, help="dir containing reranker_lift_*.json")
    ap.add_argument("--webfaq-min-delta", type=float, default=RE.WEBFAQ_MIN_DELTA)
    ap.add_argument("--local-rag-min-delta", type=float, default=RE.LOCAL_RAG_MIN_DELTA)
    ap.add_argument("--catastrophic-degradation", type=float, default=RE.CATASTROPHIC)
    ap.add_argument("--min-first-stage-recall", type=float, default=RE.MIN_FIRST_STAGE_RECALL)
    ap.add_argument("--output", default=None)
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()

    eval_dir = pathlib.Path(args.eval_dir)
    reports = []
    for p in sorted(eval_dir.glob("reranker_lift_*.json")):
        try:
            reports.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    if not reports:
        print(f"ERROR: no reranker_lift_*.json in {eval_dir}", file=sys.stderr)
        return 2

    result = RE.evaluate_promotion(reports, webfaq_min=args.webfaq_min_delta,
                                   local_min=args.local_rag_min_delta,
                                   catastrophic=args.catastrophic_degradation,
                                   min_first_stage_recall=args.min_first_stage_recall)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    verdict = ("PASS — RAG reranker may be promoted" if result["status"] == "pass"
               else "FAIL — do NOT promote")
    print(f"\n{verdict}. deltas={result['deltas']} diagnostic={result['diagnostic_sets']}")

    if args.output:
        pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    if args.markdown:
        lines = ["# v4 RAG reranker promotion gate", "", f"Status: **{result['status'].upper()}**",
                 "", "| eval set | delta_ndcg@10 | diagnostic |", "|---|--:|:--:|"]
        for r in reports:
            lines.append(f"| {r['eval_set']} | {r['delta_ndcg@10']:+} | "
                         f"{'yes' if r.get('diagnostic') else ''} |")
        if result["failing"]:
            lines += ["", "**Failing:**"] + [f"- ❌ {c['check']}: {c['detail']}" for c in result["failing"]]
        pathlib.Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
