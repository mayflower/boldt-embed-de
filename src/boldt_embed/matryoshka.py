"""Matryoshka truncation utilities (pure stdlib).

Matryoshka reduces *downstream vector size / storage cost* by using a prefix of the
embedding; it does not change the model. Truncated prefixes MUST be re-normalized before
cosine similarity, which ``truncate_normalized`` does.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from .pooling import l2_normalize

Vector = List[float]


def truncate(vec: Sequence[float], dim: int) -> Vector:
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if dim > len(vec):
        raise ValueError(f"cannot truncate length-{len(vec)} vector to {dim}")
    return list(vec[:dim])


def truncate_normalized(vec: Sequence[float], dim: int) -> Vector:
    return l2_normalize(truncate(vec, dim))


def matryoshka_views(vec: Sequence[float], dims: Sequence[int]) -> Dict[int, Vector]:
    """Return {dim: L2-normalized prefix} for each requested dim."""
    return {d: truncate_normalized(vec, d) for d in dims}
