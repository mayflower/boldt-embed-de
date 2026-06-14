"""Evaluate a RAG reranker as LIFT over FIXED candidate lists + gate promotion (pure stdlib).

A RAG reranker is judged on whether it promotes answer-supporting passages inside a fixed top-k
set. This computes first-stage vs reranked metrics per list, aggregates per eval set, and gates
promotion on RAG usefulness — never on legal (GerDaLIR is diagnostic-only).

The metric/lift/gate layer is pure stdlib (testable on fixtures with precomputed scores); the
actual reranker scoring lazy-imports torch in the CLI.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .metrics import metrics_for_query
from .rag_eval_schema import K, rag_metrics_for_query
from .reranker_modern import oracle_metrics

# --- gate defaults ---
WEBFAQ_MIN_DELTA = 0.03
LOCAL_RAG_MIN_DELTA = 0.03
NEUTRAL_MIN_DELTA = 0.0           # germanquad / dt_test must be neutral-or-better
CATASTROPHIC = -0.02              # no eval set may drop more than this
MIN_FIRST_STAGE_RECALL = 0.5      # first-stage positive_in_top_10 for reranking to be meaningful
DIAGNOSTIC_SETS = frozenset({"gerdalir", "legal"})   # reported, never a gate


def _first_stage_order(cands: Sequence[Dict[str, Any]]) -> List[str]:
    if cands and all(c.get("first_stage_score") is not None for c in cands):
        return [c["doc_id"] for c in sorted(cands, key=lambda c: c["first_stage_score"], reverse=True)]
    if cands and all(c.get("first_stage_rank") is not None for c in cands):
        return [c["doc_id"] for c in sorted(cands, key=lambda c: c["first_stage_rank"])]
    return [c["doc_id"] for c in cands]               # fall back to input order


def _reranked_order(cands: Sequence[Dict[str, Any]], scores: Dict[str, float]) -> List[str]:
    return [c["doc_id"] for c in sorted(cands, key=lambda c: scores.get(c["doc_id"], float("-inf")),
                                        reverse=True)]


def build_lift_report(rows: Sequence[Dict[str, Any]], eval_set: str,
                      scores_by_query: Optional[Dict[str, Dict[str, float]]] = None,
                      diagnostic: Optional[bool] = None) -> Dict[str, Any]:
    """First-stage vs reranked aggregate over fixed candidate lists. ``scores_by_query`` maps
    query_id -> {doc_id: reranker_score}; if absent, falls back to per-candidate ``reranker_score``
    then ``teacher_score`` (a no-op rerank if none, so delta=0)."""
    scores_by_query = scores_by_query or {}
    fs_rows, rr_rows, orc_rows = [], [], []
    fs_recall_hits = 0
    n_no_candidates = 0
    n = 0
    for r in rows:
        cands = r.get("candidates") or []
        if len(cands) < 1:
            n_no_candidates += 1
            continue
        positives = set(r.get("positive_doc_ids") or [])
        ras = bool((r.get("metadata") or {}).get("requires_answer_support"))
        per_doc = dict(scores_by_query.get(str(r.get("query_id")), {}))
        for c in cands:                               # fallbacks: reranker_score, then teacher_score
            if c["doc_id"] not in per_doc:
                s = c.get("reranker_score")
                if s is None:
                    s = c.get("teacher_score")
                if s is not None:
                    per_doc[c["doc_id"]] = float(s)
        fs_order = _first_stage_order(cands)
        rr_order = _reranked_order(cands, per_doc) if per_doc else list(fs_order)
        fs_rows.append(rag_metrics_for_query(fs_order, positives, ras))
        rr_rows.append(rag_metrics_for_query(rr_order, positives, ras))
        orc_rows.append(oracle_metrics([c["doc_id"] for c in cands], positives, (K,)))
        if set(fs_order[:K]) & positives:
            fs_recall_hits += 1
        n += 1

    def _agg(rows_, key):
        return round(sum(x.get(key, 0.0) for x in rows_) / len(rows_), 4) if rows_ else 0.0

    def _asup(rows_):
        xs = [x["answer_support_at_10"] for x in rows_ if "answer_support_at_10" in x]
        return round(sum(xs) / len(xs), 4) if xs else None

    fs_ndcg, rr_ndcg = _agg(fs_rows, "ndcg@10"), _agg(rr_rows, "ndcg@10")
    return {
        "eval_set": eval_set,
        "diagnostic": DIAGNOSTIC_SETS.__contains__(eval_set) if diagnostic is None else diagnostic,
        "fixed_candidates": n_no_candidates == 0 and n > 0,
        "n_queries": n, "n_queries_without_candidates": n_no_candidates,
        "first_stage_ndcg@10": fs_ndcg, "reranked_ndcg@10": rr_ndcg,
        "delta_ndcg@10": round(rr_ndcg - fs_ndcg, 4),
        "first_stage_mrr@10": _agg(fs_rows, "mrr@10"), "reranked_mrr@10": _agg(rr_rows, "mrr@10"),
        "positive_in_top_10_before": _agg(fs_rows, "positive_in_top_10"),
        "positive_in_top_10_after": _agg(rr_rows, "positive_in_top_10"),
        "answer_support_at_10": _asup(rr_rows),
        "oracle_ndcg@10": _agg(orc_rows, "ndcg@10"),
        "first_stage_recall_top_10": round(fs_recall_hits / n, 4) if n else 0.0,
    }


def evaluate_promotion(reports: Sequence[Dict[str, Any]], *,
                       webfaq_min: float = WEBFAQ_MIN_DELTA, local_min: float = LOCAL_RAG_MIN_DELTA,
                       neutral_min: float = NEUTRAL_MIN_DELTA, catastrophic: float = CATASTROPHIC,
                       min_first_stage_recall: float = MIN_FIRST_STAGE_RECALL) -> Dict[str, Any]:
    """Gate promotion on RAG usefulness. GerDaLIR/legal are diagnostic — never gate."""
    by_set = {r["eval_set"]: r for r in reports}
    checks: List[Dict[str, Any]] = []

    def _chk(name, passed, detail):
        checks.append({"check": name, "status": "pass" if passed else "fail", "detail": detail})

    def _delta(name):
        return by_set[name]["delta_ndcg@10"] if name in by_set else None

    # required RAG lift
    if "webfaq" in by_set:
        _chk("webfaq_delta", _delta("webfaq") >= webfaq_min, f"{_delta('webfaq')} (min {webfaq_min})")
        _chk("webfaq_first_stage_recall",
             by_set["webfaq"]["first_stage_recall_top_10"] >= min_first_stage_recall,
             f"{by_set['webfaq']['first_stage_recall_top_10']} (min {min_first_stage_recall})")
    else:
        _chk("webfaq_present", False, "WebFAQ held-out lift report is required")
    if "local_rag" in by_set:
        _chk("local_rag_delta", _delta("local_rag") >= local_min, f"{_delta('local_rag')} (min {local_min})")
        _chk("local_rag_first_stage_recall",
             by_set["local_rag"]["first_stage_recall_top_10"] >= min_first_stage_recall,
             f"{by_set['local_rag']['first_stage_recall_top_10']} (min {min_first_stage_recall})")
    # neutral-or-better on public benchmarks
    for s in ("germanquad", "dt_test"):
        if s in by_set:
            _chk(f"{s}_neutral_or_better", _delta(s) >= neutral_min, f"{_delta(s)} (min {neutral_min})")
    # no catastrophic degradation, and every reported set uses fixed candidate lists (non-diagnostic)
    for r in reports:
        if r.get("diagnostic"):
            continue
        _chk(f"{r['eval_set']}_not_catastrophic", r["delta_ndcg@10"] >= catastrophic,
             f"{r['delta_ndcg@10']} (min {catastrophic})")
        _chk(f"{r['eval_set']}_fixed_candidates", bool(r.get("fixed_candidates")),
             "all eval sets must use fixed candidate lists")

    failing = [c for c in checks if c["status"] == "fail"]
    return {"status": "fail" if failing else "pass", "checks": checks, "failing": failing,
            "diagnostic_sets": [r["eval_set"] for r in reports if r.get("diagnostic")],
            "deltas": {r["eval_set"]: r["delta_ndcg@10"] for r in reports}}
