"""Hybrid retrieval evaluation: BM25 + dense + RRF fusion + reranker + Matryoshka sweep.

Production German retrieval wants sparse *and* dense: BM25 catches exact terms (legal refs,
compounds, rare entities) that dense models smear, while dense catches paraphrase. This
module provides the fusion + metric + Matryoshka plumbing as **pure stdlib** (BM25 and cosine
reuse `eval_harness`); only dense *encoding* needs a model and is lazy-imported by the script.

Modes: ``bm25_only``, ``dense_only``, ``hybrid_rrf``, ``hybrid_rrf_plus_reranker``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .eval_harness import bm25_rank, cosine_rank
from .matryoshka import truncate_normalized
from .metrics import aggregate, metrics_for_query

DEFAULT_KS = (10, 100)


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> List[str]:
    """Fuse ranked lists by RRF: score(d) = sum_r 1/(k + rank_r(d)). Higher is better.
    Deterministic: ties broken by first appearance order across the input rankings."""
    scores: Dict[str, float] = {}
    first_seen: Dict[str, int] = {}
    order = 0
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
            if doc not in first_seen:
                first_seen[doc] = order
                order += 1
    return sorted(scores, key=lambda d: (-scores[d], first_seen[d]))


def build_candidate_set(ranking: Sequence[str], top_k: int) -> List[str]:
    return list(ranking[:top_k])


def evaluate_ranking(ranked: Sequence[str], positive_ids, ks: Sequence[int] = DEFAULT_KS
                     ) -> Dict[str, float]:
    return metrics_for_query(list(ranked), set(positive_ids), tuple(ks))


def positive_in_top_k(ranked: Sequence[str], positive_ids, k: int) -> float:
    return 1.0 if set(ranked[:k]) & set(positive_ids) else 0.0


def aggregate_over_queries(per_query: Sequence[Dict[str, float]]) -> Dict[str, float]:
    return aggregate(list(per_query))


def fuse_and_rerank(bm25_ranking: Sequence[str], dense_ranking: Sequence[str], mode: str,
                    top_k_first_stage: int = 200, top_k_rerank: int = 50,
                    rerank_fn: Optional[Callable[[Sequence[str]], List[str]]] = None,
                    rrf_k: int = 60) -> List[str]:
    """Produce the final ranking for one query under the given mode."""
    bm = build_candidate_set(bm25_ranking, top_k_first_stage)
    dn = build_candidate_set(dense_ranking, top_k_first_stage)
    if mode == "bm25_only":
        return bm
    if mode == "dense_only":
        return dn
    fused = reciprocal_rank_fusion([bm, dn], k=rrf_k)
    if mode == "hybrid_rrf":
        return fused
    if mode == "hybrid_rrf_plus_reranker":
        if rerank_fn is None:
            return fused  # no reranker available -> fall back to fusion
        head = fused[:top_k_rerank]
        reranked = rerank_fn(head)
        return reranked + fused[top_k_rerank:]
    raise ValueError(f"unknown mode '{mode}'")


def evaluate_mode(queries: Sequence[Dict[str, Any]], bm25_rankings: Dict[str, List[str]],
                  dense_rankings: Dict[str, List[str]], mode: str, *,
                  top_k_first_stage: int = 200, top_k_rerank: int = 50,
                  rerank_fns: Optional[Dict[str, Callable]] = None,
                  ks: Sequence[int] = DEFAULT_KS) -> Dict[str, float]:
    """Aggregate metrics for one mode across all queries. `queries`:[{query_id, positive_ids}]."""
    rows = []
    for q in queries:
        qid = str(q["query_id"])
        final = fuse_and_rerank(
            bm25_rankings.get(qid, []), dense_rankings.get(qid, []), mode,
            top_k_first_stage=top_k_first_stage, top_k_rerank=top_k_rerank,
            rerank_fn=(rerank_fns or {}).get(qid))
        m = evaluate_ranking(final, q["positive_ids"], ks)
        m[f"pos_in_top_{ks[0]}"] = positive_in_top_k(final, q["positive_ids"], ks[0])
        rows.append(m)
    return aggregate_over_queries(rows)


def matryoshka_sweep(query_vecs: Dict[str, Sequence[float]],
                     doc_vecs: Sequence[Tuple[str, Sequence[float]]],
                     queries: Sequence[Dict[str, Any]], dims: Sequence[int],
                     ks: Sequence[int] = DEFAULT_KS) -> Dict[int, Dict[str, float]]:
    """For each Matryoshka dim: truncate + re-normalize all vectors, dense-rank, and report
    aggregated metrics. Pure stdlib (works on plain float lists)."""
    out: Dict[int, Dict[str, float]] = {}
    for dim in dims:
        td = [(did, truncate_normalized(v, dim)) for did, v in doc_vecs]
        rows = []
        for q in queries:
            qid = str(q["query_id"])
            if qid not in query_vecs:
                continue
            ranked = cosine_rank(truncate_normalized(query_vecs[qid], dim), td)
            rows.append(evaluate_ranking(ranked, q["positive_ids"], ks))
        out[dim] = aggregate_over_queries(rows)
    return out


def bm25_rankings_for_queries(queries: Sequence[Dict[str, Any]],
                              corpus: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    return {str(q["query_id"]): bm25_rank(q["query"], corpus) for q in queries}
