#!/usr/bin/env python3
"""Evaluate the v6 reranker as the ACTUAL PRODUCT: RAW lift over FIXED candidate lists. No bounded
policy, no serving wrapper, no abstention. The ranking_mode of every report is hard-set to ``raw``
(the gate rejects anything else). Pure-stdlib metric core (testable); model scoring is lazy ML.

Per query: first-stage vs raw-reranked nDCG@10, delta, MRR@10, catastrophic-drop rate, the
candidate-set-unchanged sanity check (a reranker only reorders a fixed list), positive_present_rate,
and metrics by hardness bucket.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed.hardness_aware_eval import assign_bucket  # noqa: E402
from boldt_embed.metrics import mrr_at_k, ndcg_at_k, recall_at_k  # noqa: E402

K = 10
RANKING_MODE = "raw"          # this script ONLY evaluates the raw reranker — never a policy
PRIMARY = {"webfaq", "local_rag"}
GUARDRAIL = {"germanquad", "dt_test"}
DIAGNOSTIC = {"gerdalir", "legal"}


def _positives(row):
    pos = set(row.get("positive_doc_ids") or [])
    if row.get("positive_doc_id"):
        pos.add(row["positive_doc_id"])
    pos |= {c.get("doc_id") for c in (row.get("candidates") or []) if c.get("is_positive")}
    pos |= {c.get("doc_id") for c in (row.get("candidates") or []) if c.get("label") == 1}
    return {p for p in pos if p}


def _first_stage_order(cands):
    if cands and all(c.get("first_stage_rank") is not None for c in cands):
        return [c["doc_id"] for c in sorted(cands, key=lambda c: float(c["first_stage_rank"]))]
    if cands and all(c.get("first_stage_score") is not None for c in cands):
        return [c["doc_id"] for c in sorted(cands, key=lambda c: -float(c["first_stage_score"]))]
    return [c.get("doc_id") for c in cands]


def role_of(name: str) -> str:
    if name in DIAGNOSTIC:
        return "diagnostic"
    if name in GUARDRAIL:
        return "guardrail"
    return "primary"


def raw_lift_report(rows, scores_by_qid, name, *, ks=(10,)) -> dict:
    """Pure-stdlib raw-reranker lift report. ``scores_by_qid[qid]`` is the reranker score per
    candidate, aligned to ``row['candidates']`` order."""
    fs_n, rr_n, mrr_n = [], [], []
    cat = 0
    present = 0
    set_changed = 0
    buckets = {}
    rec10_fs = rec10_rr = 0
    n = 0
    for r in rows:
        cands = r.get("candidates") or []
        if not cands:
            continue
        n += 1
        pos = _positives(r)
        fs_ids = _first_stage_order(cands)
        scores = scores_by_qid.get(str(r.get("query_id"))) if scores_by_qid else None
        if scores is None or len(scores) != len(cands):
            # no reranker score => raw ranking is the first stage (no-op); keeps the eval honest
            rr_ids = list(fs_ids)
        else:
            order = sorted(range(len(cands)), key=lambda i: (-float(scores[i]), str(cands[i]["doc_id"])))
            rr_ids = [cands[i]["doc_id"] for i in order]
        if set(fs_ids) != set(rr_ids):
            set_changed += 1            # must never happen: a reranker only reorders a fixed list
        f = ndcg_at_k(fs_ids, pos, K); rr = ndcg_at_k(rr_ids, pos, K)
        fs_n.append(f); rr_n.append(rr); mrr_n.append(mrr_at_k(rr_ids, pos, K))
        if rr - f <= -0.2:
            cat += 1
        if pos & set(cands_ids := {c["doc_id"] for c in cands}):
            present += 1
        rec10_fs += 1 if recall_at_k(fs_ids, pos, K) > 0 else 0
        rec10_rr += 1 if recall_at_k(rr_ids, pos, K) > 0 else 0
        oracle = ndcg_at_k([d for d in fs_ids if d in pos] + [d for d in fs_ids if d not in pos], pos, K)
        buckets.setdefault(assign_bucket(f, oracle), []).append((f, rr))
    nn = n or 1

    def mean(xs):
        return round(sum(xs) / len(xs), 6) if xs else 0.0
    by_bucket = {b: {"n": len(v), "delta": round(sum(rr - f for f, rr in v) / len(v), 6)}
                 for b, v in sorted(buckets.items())}
    return {
        "eval_set": name, "role": role_of(name), "ranking_mode": RANKING_MODE, "n_queries": n,
        "first_stage_ndcg@10": mean(fs_n), "raw_reranker_ndcg@10": mean(rr_n),
        "delta_ndcg@10": round(mean(rr_n) - mean(fs_n), 6),
        "mrr@10": mean(mrr_n),
        "catastrophic_drop_rate": round(cat / nn, 6),
        "positive_present_rate": round(present / nn, 6),
        "recall@10_first_stage": round(rec10_fs / nn, 6),
        "recall@10_reranked": round(rec10_rr / nn, 6),
        "candidate_set_unchanged": set_changed == 0,
        "candidate_set_changed_queries": set_changed,
        "by_hardness_bucket": by_bucket,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reranker", required=True)
    ap.add_argument("--candidate-lists", required=True)
    ap.add_argument("--eval-set", default=None, help="set name; inferred from filename if omitted")
    ap.add_argument("--reranker-config", default=str(ROOT / "configs" / "training_reranker.json"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    rows = [json.loads(l) for l in pathlib.Path(args.candidate_lists).read_text(
        encoding="utf-8").split("\n") if l.strip()]
    name = args.eval_set or pathlib.Path(args.candidate_lists).stem.split("_")[0]

    # ---- score RAW with the trained reranker (lazy ML) ----
    from boldt_embed.config import load_reranker_config
    from boldt_embed.reranker_modern import score_with_student_reranker
    cfg = load_reranker_config(args.reranker_config)
    scores_by_qid = {}
    for r in rows:
        cands = r.get("candidates") or []
        pairs = [(r.get("query", ""), c.get("text") or c.get("document", "")) for c in cands]
        if pairs:
            scores_by_qid[str(r.get("query_id"))] = score_with_student_reranker(
                args.reranker, pairs, cfg.input_template, max_length=args.max_length)

    rep = raw_lift_report(rows, scores_by_qid, name)
    rep["reranker"] = args.reranker
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"[v6-raw-eval] {name} ({rep['role']}): fs {rep['first_stage_ndcg@10']} -> raw "
          f"{rep['raw_reranker_ndcg@10']} (Δ {rep['delta_ndcg@10']:+}) mode {rep['ranking_mode']} "
          f"catastrophic {rep['catastrophic_drop_rate']} present {rep['positive_present_rate']} "
          f"-> {args.output}")
    if not rep["candidate_set_unchanged"]:
        print(f"WARNING: reranking changed the candidate set on {rep['candidate_set_changed_queries']} "
              "queries — a raw reranker must only REORDER a fixed list", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
