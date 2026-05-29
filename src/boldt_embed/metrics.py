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


# --------------------------------------------------------------- STS / classification / clustering
def _ranks(xs: Sequence[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation (for STS). 0.0 if degenerate."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def accuracy(true: Sequence, pred: Sequence) -> float:
    if not true:
        return 0.0
    return sum(1 for t, p in zip(true, pred) if t == p) / len(true)


def _entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counts if c > 0)


def v_measure(true: Sequence, pred: Sequence) -> float:
    """V-measure (harmonic mean of homogeneity and completeness) for clustering."""
    from collections import defaultdict

    n = len(true)
    if n == 0:
        return 0.0
    ct: Dict = defaultdict(int)
    cp: Dict = defaultdict(int)
    joint: Dict = defaultdict(int)
    for t, p in zip(true, pred):
        ct[t] += 1
        cp[p] += 1
        joint[(t, p)] += 1
    h_c = _entropy(list(ct.values()))
    h_k = _entropy(list(cp.values()))
    h_c_given_k = -sum((cnt / n) * math.log(cnt / cp[p]) for (t, p), cnt in joint.items() if cnt > 0)
    h_k_given_c = -sum((cnt / n) * math.log(cnt / ct[t]) for (t, p), cnt in joint.items() if cnt > 0)
    homogeneity = 1.0 if h_c == 0 else 1 - h_c_given_k / h_c
    completeness = 1.0 if h_k == 0 else 1 - h_k_given_c / h_k
    if homogeneity + completeness == 0:
        return 0.0
    return 2 * homogeneity * completeness / (homogeneity + completeness)
