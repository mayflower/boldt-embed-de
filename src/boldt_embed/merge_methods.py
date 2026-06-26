"""Pure-stdlib checkpoint-merging math for the v8 specialist→merge lever (Prompt 08).

This module is the math half of the merge search; ``scripts/ar_merge_search.py`` is the IO
half. The contract is deliberately tiny and torch-free so it is unit-testable on CPU with no
weights: every function operates on **state dicts represented as ``Dict[str, list[float]]``** —
each parameter tensor flattened to a plain Python list of floats. The orchestrator flattens the
real tensors, calls into here, then reshapes the results back.

Hard rules (fail-closed):
  * All parents must share the SAME keys and EQUAL per-key lengths, else ``ValueError``.
  * Task-arithmetic methods (``task_vector_sum``, ``ties``, ``dare_linear``) operate relative to
    a ``base`` (the common warm-start). The orchestrator only offers these when a ``warm_start``
    is configured; without one it marks them ``unsupported`` rather than mis-merging.

NO ``torch``/``numpy`` import here — importing this module must stay dependency-free.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Sequence

StateDict = Dict[str, List[float]]

# Methods this module can compute. Whether a given method is *applicable* (e.g. needs a base)
# is decided by the orchestrator; this is just the math vocabulary.
METHODS = (
    "mean",
    "weighted_mean",
    "slerp_pairwise",
    "task_vector_sum",
    "ties",
    "dare_linear",
    "layerwise_weighted_mean",
)


# --------------------------------------------------------------------------------------------
# shared validation
# --------------------------------------------------------------------------------------------
def _check_parents(parents: Sequence[StateDict]) -> List[str]:
    """Assert >=1 parent, identical key sets and equal per-key lengths. Return the key list."""
    if not parents:
        raise ValueError("no parents provided")
    keys = list(parents[0].keys())
    key_set = set(keys)
    for i, sd in enumerate(parents):
        if set(sd.keys()) != key_set:
            missing = key_set ^ set(sd.keys())
            raise ValueError(f"parent {i} has mismatched keys (differing: {sorted(missing)})")
        for k in keys:
            if len(sd[k]) != len(parents[0][k]):
                raise ValueError(
                    f"length mismatch on key {k!r}: parent 0 has {len(parents[0][k])}, "
                    f"parent {i} has {len(sd[k])}"
                )
    return keys


def _check_base(base: StateDict, parents: Sequence[StateDict], keys: Sequence[str]) -> None:
    """Assert the base shares the parents' keys + per-key lengths."""
    if set(base.keys()) != set(keys):
        missing = set(keys) ^ set(base.keys())
        raise ValueError(f"base has mismatched keys (differing: {sorted(missing)})")
    for k in keys:
        if len(base[k]) != len(parents[0][k]):
            raise ValueError(
                f"base length mismatch on key {k!r}: base has {len(base[k])}, "
                f"parents have {len(parents[0][k])}"
            )


def _normalize_weights(weights: Sequence[float], n: int) -> List[float]:
    if len(weights) != n:
        raise ValueError(f"expected {n} weights, got {len(weights)}")
    if any(float(w) < 0.0 for w in weights):
        raise ValueError(f"weights must be non-negative (got {list(weights)}); a negative weight is "
                         "an extrapolated, non-convex merge and is rejected fail-closed")
    s = float(sum(weights))
    if s == 0.0:
        raise ValueError("weights sum to zero")
    return [float(w) / s for w in weights]


# --------------------------------------------------------------------------------------------
# vector helpers (single flattened tensor)
# --------------------------------------------------------------------------------------------
def _vec_weighted_sum(vectors: Sequence[Sequence[float]], weights: Sequence[float]) -> List[float]:
    n = len(vectors[0])
    acc = [0.0] * n
    for w, v in zip(weights, vectors):
        for j in range(n):
            acc[j] += w * v[j]
    return acc


def _slerp_vec(a: Sequence[float], b: Sequence[float], t: float, eps: float = 1e-8) -> List[float]:
    """Spherical interpolation between two flattened tensors; LERP fallback when near-parallel."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < eps or nb < eps:
        return [(1.0 - t) * x + t * y for x, y in zip(a, b)]
    cos_omega = max(-1.0, min(1.0, dot / (na * nb)))
    omega = math.acos(cos_omega)
    sin_omega = math.sin(omega)
    if sin_omega < 1e-6:  # nearly colinear → plain LERP
        return [(1.0 - t) * x + t * y for x, y in zip(a, b)]
    c1 = math.sin((1.0 - t) * omega) / sin_omega
    c2 = math.sin(t * omega) / sin_omega
    return [c1 * x + c2 * y for x, y in zip(a, b)]


def _seeded_mask(seed: int, key: str, length: int, density: float) -> List[bool]:
    """Deterministic keep-mask of ~``density`` True entries, derived from a seeded index hash.

    No ``random`` module / no global state: each position's pseudo-random value is a stable hash of
    (seed, key, index) mapped into [0, 1). A position is kept when its value < density. The key is
    folded in via a STABLE sha256 digest (NOT the builtin ``hash()``, which is salted per-process by
    PYTHONHASHSEED) so the mask — and therefore a DARE merge — is reproducible ACROSS processes.
    """
    key_h = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")
    keep: List[bool] = []
    for i in range(length):
        # splitmix64-style avalanche over a composed 64-bit key — pure integer arithmetic.
        h = (seed & 0xFFFFFFFFFFFFFFFF)
        h = (h * 0x9E3779B97F4A7C15 + (i + 1) + key_h) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 30
        h = (h * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 27
        h = (h * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 31
        r = (h & 0xFFFFFFFFFFFFFFFF) / float(1 << 64)  # in [0, 1)
        keep.append(r < density)
    return keep


# --------------------------------------------------------------------------------------------
# merge methods (state-dict in, state-dict out)
# --------------------------------------------------------------------------------------------
def mean(parents: Sequence[StateDict]) -> StateDict:
    """Uniform model soup (Wortsman 2022)."""
    keys = _check_parents(parents)
    w = [1.0 / len(parents)] * len(parents)
    return {k: _vec_weighted_sum([p[k] for p in parents], w) for k in keys}


def weighted_mean(parents: Sequence[StateDict], weights: Sequence[float]) -> StateDict:
    """Weighted model soup. Weights are normalized to sum to 1."""
    keys = _check_parents(parents)
    w = _normalize_weights(weights, len(parents))
    return {k: _vec_weighted_sum([p[k] for p in parents], w) for k in keys}


def slerp_pairwise(parents: Sequence[StateDict], t: float) -> StateDict:
    """Spherical interpolation between EXACTLY two parents (per-tensor)."""
    keys = _check_parents(parents)
    if len(parents) != 2:
        raise ValueError(f"slerp_pairwise requires exactly 2 parents, got {len(parents)}")
    a, b = parents[0], parents[1]
    return {k: _slerp_vec(a[k], b[k], t) for k in keys}


def task_vector_sum(parents: Sequence[StateDict], base: StateDict) -> StateDict:
    """Task arithmetic (Ilharco 2023): sum each parent's delta from the base, add back to base."""
    keys = _check_parents(parents)
    _check_base(base, parents, keys)
    out: StateDict = {}
    for k in keys:
        b = base[k]
        acc = list(b)
        for p in parents:
            for j in range(len(b)):
                acc[j] += p[k][j] - b[j]
        out[k] = acc
    return out


def ties(parents: Sequence[StateDict], base: StateDict, density: float) -> StateDict:
    """TIES-Merging (Yadav 2023) per parameter index, relative to the warm-start base.

    1. TRIM: for each parent, keep only the top ``density`` fraction of its task-vector entries by
       magnitude (the rest zeroed) — done globally per key.
    2. ELECT SIGN: per index, the aggregate sign is the sign of the summed (trimmed) deltas.
    3. DISJOINT MERGE: average only the entries whose sign agrees with the elected sign.
    The merged delta is added back to the base.
    """
    keys = _check_parents(parents)
    _check_base(base, parents, keys)
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density must be in (0, 1], got {density}")
    out: StateDict = {}
    for k in keys:
        b = base[k]
        n = len(b)
        # task vectors (deltas) per parent
        deltas = [[p[k][j] - b[j] for j in range(n)] for p in parents]
        # TRIM each delta to top-density by magnitude (per key)
        keep_count = max(1, int(round(density * n)))
        trimmed = []
        for d in deltas:
            order = sorted(range(n), key=lambda idx: abs(d[idx]), reverse=True)
            kept = set(order[:keep_count])
            trimmed.append([d[j] if j in kept else 0.0 for j in range(n)])
        # ELECT SIGN per index from the summed trimmed deltas
        merged_delta = [0.0] * n
        for j in range(n):
            total = sum(t[j] for t in trimmed)
            elected = 1.0 if total > 0 else (-1.0 if total < 0 else 0.0)
            if elected == 0.0:
                merged_delta[j] = 0.0
                continue
            # DISJOINT MERGE: mean of entries agreeing with the elected sign
            agree = [t[j] for t in trimmed if (t[j] > 0 and elected > 0) or (t[j] < 0 and elected < 0)]
            merged_delta[j] = sum(agree) / len(agree) if agree else 0.0
        out[k] = [b[j] + merged_delta[j] for j in range(n)]
    return out


def dare_linear(
    parents: Sequence[StateDict],
    base: StateDict,
    density: float,
    weights: Sequence[float] | None = None,
    rescale: bool = True,
    seed: int = 0,
) -> StateDict:
    """DARE (Yu 2024): Drop And REscale each parent's task vector, then linearly combine.

    Per parent task vector: drop entries to ``density`` survival probability via a DETERMINISTIC
    seeded mask, then (if ``rescale``) divide survivors by ``density`` so the expected magnitude is
    preserved. The rescaled deltas are weighted-combined and added back to the base.
    """
    keys = _check_parents(parents)
    _check_base(base, parents, keys)
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density must be in (0, 1], got {density}")
    w = _normalize_weights(weights, len(parents)) if weights is not None else \
        [1.0 / len(parents)] * len(parents)
    scale = (1.0 / density) if rescale else 1.0
    out: StateDict = {}
    for k in keys:
        b = base[k]
        n = len(b)
        merged_delta = [0.0] * n
        for pi, p in enumerate(parents):
            # distinct mask per parent: fold the parent index into the seed
            mask = _seeded_mask(seed + pi * 0x1000193, k, n, density)
            for j in range(n):
                if mask[j]:
                    delta = (p[k][j] - b[j]) * scale
                    merged_delta[j] += w[pi] * delta
        out[k] = [b[j] + merged_delta[j] for j in range(n)]
    return out


def layerwise_weighted_mean(
    parents: Sequence[StateDict], weights_per_key: Dict[str, Sequence[float]]
) -> StateDict:
    """Weighted mean with a distinct weight vector per key (per-layer soup).

    ``weights_per_key`` maps each key to a list of per-parent weights (normalized per key). Keys
    absent from the mapping fall back to a uniform mean.
    """
    keys = _check_parents(parents)
    out: StateDict = {}
    for k in keys:
        if k in weights_per_key:
            w = _normalize_weights(weights_per_key[k], len(parents))
        else:
            w = [1.0 / len(parents)] * len(parents)
        out[k] = _vec_weighted_sum([p[k] for p in parents], w)
    return out
