"""Hardness-aware RAG reranker evaluation (pure stdlib, no ML).

v4 over-penalized the reranker on GermanQuAD/DT-test, whose first stages were near-ceiling (high
recall, oracle ~1.0): a reranker can only churn an already-correct top order there, so a tiny
negative delta is noise, not failure. v5 instead:

- buckets each candidate list by **difficulty** (no_room / easy / medium / hard / impossible);
- drives promotion from **medium+hard** buckets on WebFAQ/local/private RAG (where there is real
  headroom);
- treats GermanQuAD/DT-test as **do-not-regress guardrails** with a small tolerance, never a
  primary signal — but still blocks real degradation (per-query catastrophic drops, or a guardrail
  falling below tolerance).

Candidate-list row shape (compatible with v4/webfaq2): ``{query_id, candidates: [{doc_id,
first_stage_rank|first_stage_score, reranker_score|teacher_score, is_positive, source}], ...}`` with
positives from ``positive_doc_ids`` / ``positive_doc_id`` / ``is_positive``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .metrics import ndcg_at_k

K = 10
PRIMARY_BUCKETS = ("medium", "hard")
NO_ROOM_TOLERANCE = -0.005          # near-ceiling sets: do-not-regress within a small noise band
GUARDRAIL_TOLERANCE = 0.0           # guardrails WITH real headroom must be neutral-or-better
CATASTROPHIC_PER_QUERY_DROP = 0.2   # a single-query delta <= -this is a catastrophic drop
MAX_CATASTROPHIC_RATE = 0.05        # fraction of queries allowed to drop catastrophically
NO_ROOM_MAJORITY = 0.5              # >= this fraction in no_room => "mostly near-ceiling"
PRIMARY_MIN_LIFT = 0.0              # medium+hard lift must exceed this (default: strictly positive)


def assign_bucket(first_stage_ndcg: float, oracle_ndcg: float) -> str:
    """Difficulty bucket for one candidate list. Order matters (impossible/no_room first)."""
    if oracle_ndcg < 0.80:
        return "impossible"
    if oracle_ndcg >= 0.98 and first_stage_ndcg >= 0.95:
        return "no_room"
    if first_stage_ndcg >= 0.85:
        return "easy"
    if first_stage_ndcg >= 0.50:
        return "medium"
    return "hard"                    # first_stage < 0.50 and oracle >= 0.80


def _positives(row: Dict[str, Any], cands: Sequence[Dict[str, Any]]) -> set:
    pos = set(row.get("positive_doc_ids") or [])
    if row.get("positive_doc_id"):
        pos.add(row["positive_doc_id"])
    pos |= {c["doc_id"] for c in cands if c.get("is_positive")}
    return pos


def _first_stage_order(cands: Sequence[Dict[str, Any]]) -> List[str]:
    if cands and all(c.get("first_stage_score") is not None for c in cands):
        return [c["doc_id"] for c in sorted(cands, key=lambda c: c["first_stage_score"], reverse=True)]
    if cands and all(c.get("first_stage_rank") is not None for c in cands):
        return [c["doc_id"] for c in sorted(cands, key=lambda c: c["first_stage_rank"])]
    return [c["doc_id"] for c in cands]


def _reranked_order(cands: Sequence[Dict[str, Any]], scores: Dict[str, float]) -> List[str]:
    return [c["doc_id"] for c in sorted(cands, key=lambda c: scores.get(c["doc_id"], float("-inf")),
                                        reverse=True)]


def list_metrics(row: Dict[str, Any],
                 reranker_scores: Optional[Dict[str, float]] = None) -> Optional[Dict[str, Any]]:
    """Per-query hardness + lift metrics for one fixed candidate list (None if no candidates)."""
    cands = row.get("candidates") or []
    if not cands:
        return None
    positives = _positives(row, cands)
    per_doc = dict(reranker_scores or {})
    for c in cands:                                  # fallbacks: reranker_score, then teacher_score
        if c["doc_id"] not in per_doc:
            s = c.get("reranker_score")
            if s is None:
                s = c.get("teacher_score")
            if s is not None:
                per_doc[c["doc_id"]] = float(s)

    fs_order = _first_stage_order(cands)
    rr_order = _reranked_order(cands, per_doc) if per_doc else list(fs_order)
    oracle_order = [d for d in fs_order if d in positives] + [d for d in fs_order if d not in positives]

    fs = round(ndcg_at_k(fs_order, positives, K), 6)
    rr = round(ndcg_at_k(rr_order, positives, K), 6)
    orc = round(ndcg_at_k(oracle_order, positives, K), 6)
    return {
        "query_id": row.get("query_id"),
        "first_stage_ndcg@10": fs, "reranked_ndcg@10": rr, "oracle_ndcg@10": orc,
        "delta": round(rr - fs, 6),
        "positive_in_top_10": 1.0 if set(fs_order[:10]) & positives else 0.0,
        "positive_in_top_50": 1.0 if set(fs_order[:50]) & positives else 0.0,
        "num_candidates": len(cands),
        "num_candidate_sources": len({c.get("source") for c in cands if c.get("source")}),
        "hardness_bucket": assign_bucket(fs, orc),
    }


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize_eval_set(name: str, per_query: Sequence[Dict[str, Any]], *, role: str = "primary",
                       catastrophic_per_query_drop: float = CATASTROPHIC_PER_QUERY_DROP
                       ) -> Dict[str, Any]:
    """Aggregate per-query metrics into bucketed macro/micro lift + catastrophic-drop stats."""
    pq = [m for m in per_query if m]
    n = len(pq)
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for m in pq:
        buckets.setdefault(m["hardness_bucket"], []).append(m)

    by_bucket = {b: {
        "n": len(ms),
        "mean_delta": round(_mean([m["delta"] for m in ms]), 6),
        "mean_first_stage": round(_mean([m["first_stage_ndcg@10"] for m in ms]), 6),
        "mean_reranked": round(_mean([m["reranked_ndcg@10"] for m in ms]), 6),
        "mean_oracle": round(_mean([m["oracle_ndcg@10"] for m in ms]), 6),
    } for b, ms in sorted(buckets.items())}

    primary = [m for m in pq if m["hardness_bucket"] in PRIMARY_BUCKETS]
    primary_bucket_means = [by_bucket[b]["mean_delta"] for b in PRIMARY_BUCKETS if b in by_bucket]
    catastrophic = [m for m in pq if m["delta"] <= -catastrophic_per_query_drop]
    no_room_n = len(buckets.get("no_room", []))

    return {
        "eval_set": name, "role": role, "n_queries": n,
        "bucket_counts": {b: len(ms) for b, ms in sorted(buckets.items())},
        "by_bucket": by_bucket,
        "overall_delta_ndcg@10": round(_mean([m["delta"] for m in pq]), 6),
        "micro_lift_all": round(_mean([m["delta"] for m in pq]), 6),
        "macro_lift_by_bucket": round(_mean([v["mean_delta"] for v in by_bucket.values()]), 6),
        "primary_n": len(primary),
        "primary_micro_lift": round(_mean([m["delta"] for m in primary]), 6),
        "primary_macro_lift": round(_mean(primary_bucket_means), 6),
        "no_room_fraction": round(no_room_n / n, 6) if n else 0.0,
        "mostly_no_room": (no_room_n / n >= NO_ROOM_MAJORITY) if n else False,
        "catastrophic_drops": len(catastrophic),
        "catastrophic_rate": round(len(catastrophic) / n, 6) if n else 0.0,
        "catastrophic_query_ids": [m.get("query_id") for m in catastrophic][:50],
    }


def evaluate_hardness_gate(set_reports: Sequence[Dict[str, Any]], *,
                           primary_min_lift: float = PRIMARY_MIN_LIFT,
                           no_room_tolerance: float = NO_ROOM_TOLERANCE,
                           guardrail_tolerance: float = GUARDRAIL_TOLERANCE,
                           max_catastrophic_rate: float = MAX_CATASTROPHIC_RATE) -> Dict[str, Any]:
    """Gate: medium+hard lift positive on primary sets; guardrails do-not-regress; few catastrophes."""
    checks: List[Dict[str, Any]] = []

    def _chk(name, passed, detail):
        checks.append({"check": name, "status": "pass" if passed else "fail", "detail": detail})

    has_primary = False
    for r in set_reports:
        name = r["eval_set"]
        cat_ok = r["catastrophic_rate"] <= max_catastrophic_rate + 1e-9
        _chk(f"{name}_catastrophic_rate", cat_ok,
             f"{r['catastrophic_rate']:.4f} (max {max_catastrophic_rate})")
        if r["role"] == "primary":
            has_primary = True
            if r["primary_n"] == 0:
                _chk(f"{name}_primary_lift", False,
                     "no medium+hard queries — cannot establish promotion signal")
            else:
                micro, macro = r["primary_micro_lift"], r["primary_macro_lift"]
                ok = micro > primary_min_lift and macro > primary_min_lift
                _chk(f"{name}_primary_lift", ok,
                     f"micro {micro:+.4f} / macro {macro:+.4f} medium+hard (min {primary_min_lift})")
        else:  # guardrail
            tol = no_room_tolerance if r["mostly_no_room"] else guardrail_tolerance
            ok = r["overall_delta_ndcg@10"] >= tol - 1e-9
            _chk(f"{name}_guardrail_do_not_regress", ok,
                 f"delta {r['overall_delta_ndcg@10']:+.4f} (tol {tol}; "
                 f"{'near-ceiling' if r['mostly_no_room'] else 'has-headroom'})")

    if not has_primary:
        _chk("has_primary_eval_set", False, "no primary eval set provided — cannot promote")

    failing = [c for c in checks if c["status"] == "fail"]
    return {
        "status": "pass" if not failing else "fail",
        "checks": checks, "failing": failing,
        "params": {"primary_min_lift": primary_min_lift, "no_room_tolerance": no_room_tolerance,
                   "guardrail_tolerance": guardrail_tolerance,
                   "max_catastrophic_rate": max_catastrophic_rate},
    }
