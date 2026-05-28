"""Reference checkpoint-merging math (pure stdlib, vector form).

The bidirectional track may merge the MNTP-adapted and contrastively-trained checkpoints.
These functions implement the merge math (linear / SLERP) on plain vectors so it is
unit-testable; the torch path applies the same math per parameter tensor.
"""
from __future__ import annotations

import math
from typing import List, Sequence

Vector = List[float]
METHODS = ("linear", "slerp")


def lerp(a: Sequence[float], b: Sequence[float], t: float) -> Vector:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [(1.0 - t) * x + t * y for x, y in zip(a, b)]


def slerp(a: Sequence[float], b: Sequence[float], t: float, eps: float = 1e-8) -> Vector:
    """Spherical linear interpolation; norm-preserving for unit inputs.

    Falls back to ``lerp`` when the two vectors are nearly colinear.
    """
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < eps or nb < eps:
        return lerp(a, b, t)
    cos_omega = max(-1.0, min(1.0, dot / (na * nb)))
    omega = math.acos(cos_omega)
    sin_omega = math.sin(omega)
    if sin_omega < eps:
        return lerp(a, b, t)
    c1 = math.sin((1.0 - t) * omega) / sin_omega
    c2 = math.sin(t * omega) / sin_omega
    return [c1 * x + c2 * y for x, y in zip(a, b)]


def merge(method: str, a: Sequence[float], b: Sequence[float], t: float = 0.5) -> Vector:
    if method in ("linear", "lerp"):
        return lerp(a, b, t)
    if method == "slerp":
        return slerp(a, b, t)
    raise ValueError(f"unknown merge method {method!r} (allowed: {METHODS} or 'lerp')")
