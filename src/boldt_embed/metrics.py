"""Ranking metrics for retrieval evaluation (pure stdlib).

Binary relevance. ``ranked`` is a list of doc ids best-first; ``positives`` is the set of
relevant doc ids for the query.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Set


def dcg(relevances: Sequence[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(ranked: Sequence[str], positives: Set[str], k: int) -> float:
    rels = [1 if d in positives else 0 for d in ranked[:k]]
    ideal = [1] * min(len(positives), k) + [0] * max(0, k - len(positives))
    idcg = dcg(ideal)
    return dcg(rels) / idcg if idcg else 0.0


def mrr_at_k(ranked: Sequence[str], positives: Set[str], k: int) -> float:
    for i, d in enumerate(ranked[:k]):
        if d in positives:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked: Sequence[str], positives: Set[str], k: int) -> float:
    if not positives:
        return 0.0
    return len(set(ranked[:k]) & positives) / len(positives)


def precision_at_k(ranked: Sequence[str], positives: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(ranked[:k]) & positives) / k


def average_precision_at_k(ranked: Sequence[str], positives: Set[str], k: int) -> float:
    if not positives:
        return 0.0
    hits = 0
    score = 0.0
    for i, d in enumerate(ranked[:k]):
        if d in positives:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(positives), k)


def metrics_for_query(
    ranked: Sequence[str], positives, ks: Sequence[int] = (1, 3, 5, 10)
) -> Dict[str, float]:
    pos = set(positives)
    out: Dict[str, float] = {}
    for k in ks:
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, pos, k)
        out[f"mrr@{k}"] = mrr_at_k(ranked, pos, k)
        out[f"recall@{k}"] = recall_at_k(ranked, pos, k)
        out[f"map@{k}"] = average_precision_at_k(ranked, pos, k)
    return out


def aggregate(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    n = len(rows)
    return {k: sum(r[k] for r in rows) / n for k in keys}
