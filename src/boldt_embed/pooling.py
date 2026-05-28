"""Mask-aware pooling and L2 normalization (pure stdlib, lists of floats).

These reference implementations let pooling be unit-tested without torch/numpy and are
the contract the torch model wrappers must match. A "hidden" is a list of token vectors
(``List[List[float]]``, shape [seq_len, dim]); ``mask`` is a list of 0/1 ints.
"""
from __future__ import annotations

import math
from typing import List, Sequence

Vector = List[float]
Hidden = Sequence[Sequence[float]]
Mask = Sequence[int]


def _validate(hidden: Hidden, mask: Mask) -> None:
    if len(hidden) != len(mask):
        raise ValueError(f"hidden length {len(hidden)} != mask length {len(mask)}")
    if not hidden:
        raise ValueError("hidden is empty")
    if sum(1 for m in mask if m) == 0:
        raise ValueError("mask selects no tokens")


def mean_pool(hidden: Hidden, mask: Mask) -> Vector:
    _validate(hidden, mask)
    dim = len(hidden[0])
    acc = [0.0] * dim
    count = 0
    for vec, m in zip(hidden, mask):
        if m:
            count += 1
            for j in range(dim):
                acc[j] += vec[j]
    return [a / count for a in acc]


def last_token_pool(hidden: Hidden, mask: Mask) -> Vector:
    """EOS / last non-pad token (the last index where mask == 1)."""
    _validate(hidden, mask)
    last = max(i for i, m in enumerate(mask) if m)
    return list(hidden[last])


# EOS pooling is last-non-pad pooling for right-padded sequences.
eos_pool = last_token_pool


def cls_pool(hidden: Hidden, mask: Mask) -> Vector:
    _validate(hidden, mask)
    return list(hidden[0])


_DISPATCH = {
    "mean": mean_pool,
    "eos": last_token_pool,
    "last_token": last_token_pool,
    "eos_or_last_token": last_token_pool,
    "cls": cls_pool,
}


def pool(strategy: str, hidden: Hidden, mask: Mask) -> Vector:
    if strategy not in _DISPATCH:
        raise ValueError(f"unsupported pooling strategy for reference impl: {strategy!r}")
    return _DISPATCH[strategy](hidden, mask)


def l2_normalize(vec: Sequence[float], eps: float = 1e-12) -> Vector:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < eps:
        return [0.0 for _ in vec]
    return [x / norm for x in vec]
