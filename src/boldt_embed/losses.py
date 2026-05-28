"""Reference contrastive losses (pure stdlib).

InfoNCE / MultipleNegativesRankingLoss with in-batch negatives and optional explicit
hard negatives. This is a *reference* implementation for unit-testing the training
objective's behavior without torch; the real trainer uses the framework loss.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

Vector = Sequence[float]


def _dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Vector) -> float:
    return math.sqrt(_dot(a, a))


def cosine_similarity(a: Vector, b: Vector, eps: float = 1e-12) -> float:
    na, nb = _norm(a), _norm(b)
    if na < eps or nb < eps:
        return 0.0
    return _dot(a, b) / (na * nb)


def info_nce_loss(
    queries: Sequence[Vector],
    positives: Sequence[Vector],
    hard_negatives: Optional[Sequence[Sequence[Vector]]] = None,
    temperature: float = 0.05,
) -> float:
    """Mean InfoNCE loss.

    For query i, the candidate set is all ``positives`` (in-batch negatives) plus any
    per-query ``hard_negatives[i]``. The target is ``positives[i]`` at index i.
    Lower is better; a correctly aligned, well-separated batch approaches 0.
    """
    if len(queries) != len(positives):
        raise ValueError("queries and positives must have equal length")
    if not queries:
        raise ValueError("empty batch")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if hard_negatives is not None and len(hard_negatives) != len(queries):
        raise ValueError("hard_negatives must have one entry per query")

    total = 0.0
    for i, q in enumerate(queries):
        candidates: List[Vector] = list(positives)
        if hard_negatives is not None:
            candidates = candidates + list(hard_negatives[i])
        logits = [cosine_similarity(q, c) / temperature for c in candidates]
        m = max(logits)
        log_sum_exp = m + math.log(sum(math.exp(s - m) for s in logits))
        total += log_sum_exp - logits[i]  # -log softmax at the positive index
    return total / len(queries)


# Alias used in configs / SentenceTransformers terminology.
multiple_negatives_ranking_loss = info_nce_loss
