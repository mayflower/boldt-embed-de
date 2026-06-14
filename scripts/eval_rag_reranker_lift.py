#!/usr/bin/env python3
"""Evaluate the v4 RAG reranker as LIFT over a FIXED candidate list (stdlib + lazy ML).

Scores each fixed candidate list with the reranker and reports first-stage vs reranked nDCG/MRR
@10, positive_in_top_10 before/after, answer_support_at_10, oracle, and delta_ndcg@10.
`--dry-run` imports no torch: it reranks by any precomputed reranker_score/teacher_score on the
candidates (no-op rerank if none) — for CI / wiring checks.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import rag_reranker_eval as RE  # noqa: E402


def _read(path):
    return [json.loads(l) for l in pathlib.Path(path).read_text(encoding="utf-8").split("\n")
            if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reranker", default=None, help="reranker checkpoint dir (real run)")
    ap.add_argument("--candidate-lists", required=True, help="FIXED candidate-list JSONL")
    ap.add_argument("--eval-set", default=None, help="eval set name (default: from --output stem)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "training_reranker.json"))
    ap.add_argument("--diagnostic", action="store_true", help="mark this set diagnostic-only")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = _read(args.candidate_lists)
    eval_set = args.eval_set or pathlib.Path(args.output).stem.replace("reranker_lift_", "")

    scores_by_query = None
    if not args.dry_run:
        if not args.reranker:
            print("ERROR: --reranker required for a real run", file=sys.stderr)
            return 2
        from boldt_embed import reranker_modern as RM
        from boldt_embed.config import load_reranker_config
        cfg = load_reranker_config(args.config)
        scores_by_query = {}
        for r in rows:
            cands = r.get("candidates") or []
            pairs = [(r.get("query", ""), c.get("text", "")) for c in cands]
            s = RM.score_with_student_reranker(args.reranker, pairs, cfg.input_template)
            scores_by_query[str(r.get("query_id"))] = {c["doc_id"]: sc for c, sc in zip(cands, s)}
    else:
        assert "torch" not in sys.modules, "dry-run must not import torch"

    report = RE.build_lift_report(rows, eval_set, scores_by_query=scores_by_query,
                                  diagnostic=args.diagnostic or None)
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[rag-lift] {eval_set}: first_stage {report['first_stage_ndcg@10']} -> reranked "
          f"{report['reranked_ndcg@10']} (Δ {report['delta_ndcg@10']:+}); "
          f"fixed_candidates={report['fixed_candidates']} -> {out}", file=sys.stderr)
    if args.dry_run:
        print("dry-run-ok (no ML imports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
