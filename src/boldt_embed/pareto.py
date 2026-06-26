"""Pure-stdlib Pareto-dominance logic for the AutoResearch frontier report.

This module knows nothing about disk, JSON, or MTEB — it is a small, deterministic
dominance calculator over candidates described as ``{metric_name: value_or_None}``.
``scripts/ar_report.py`` is the I/O layer that reads saved artifacts and feeds them here.

Two metric directions are supported:

  - ``HIGHER`` (the default): a larger value is better. These are the *hard target*
    metrics that decide domination — retrieval quality (nDCG/recall), Matryoshka
    retention, etc.
  - ``LOWER``: a smaller value is better — currently only ``vram_gb``.

Cost metrics (``vram_gb`` lower-better, ``throughput_pairs_per_sec`` higher-better) are,
per the prompt, *optional / tie-breakers*. By default they do NOT participate in the
dominance test (``cost_metrics`` are excluded from the objective set); they are only
used by :func:`tie_break` to order candidates that are mutually non-dominated. Callers
may opt cost metrics into the objective set explicitly if desired.

Dominance rule (documented and tested)
--------------------------------------
Candidate ``A`` *dominates* candidate ``B`` iff, over the set of objective metrics:

  1. ``A`` is **no worse than** ``B`` on **every** metric where BOTH have a value, and
  2. ``A`` is **strictly better** than ``B`` on **at least one** metric where both
     have a value.

The whole question is how to treat ``None`` (a *missing / unknown* measurement — e.g. an
MTEB task that was never evaluated for that candidate). We are deliberately
**conservative**: a missing value can neither help nor hurt a candidate.

  - A ``None`` on either side of a metric means that metric is **skipped** for that
    pairwise comparison: the candidate with the missing value cannot *claim superiority*
    on that axis (so a hole can't manufacture a win), and it cannot be *declared inferior*
    on it either (so a hole can't manufacture a loss).
  - Consequently a candidate that is missing every comparable metric can neither dominate
    nor be dominated; it survives onto the frontier as "incomparable / unknown". This is
    the safe choice for a research report: we never silently treat a missing benchmark as
    a 0 and we never let an unmeasured candidate knock a measured one off the frontier.

The non-dominated set returned by :func:`pareto_frontier` therefore contains every
candidate that no *other* candidate strictly dominates under this rule.
"""
from __future__ import annotations

# Metric direction sentinels.
HIGHER = "higher_is_better"
LOWER = "lower_is_better"

# Cost metrics: tie-breakers only, not part of the dominance objective by default.
#   vram_gb -> lower is better, throughput_pairs_per_sec -> higher is better.
DEFAULT_COST_METRICS = ("vram_gb", "throughput_pairs_per_sec")
DEFAULT_DIRECTIONS = {"vram_gb": LOWER}


def _direction(metric: str, directions: dict | None) -> str:
    if directions and metric in directions:
        return directions[metric]
    if metric in DEFAULT_DIRECTIONS:
        return DEFAULT_DIRECTIONS[metric]
    return HIGHER


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _better(a, b, direction: str) -> bool:
    """Is ``a`` strictly better than ``b`` for this direction? (both must be numbers)."""
    return a > b if direction == HIGHER else a < b


def _worse(a, b, direction: str) -> bool:
    return a < b if direction == HIGHER else a > b


def objective_metrics(candidates, cost_metrics=DEFAULT_COST_METRICS,
                      extra_objectives=None):
    """The set of metric names used for the dominance test.

    The union of all metric keys present across ``candidates``, minus the cost metrics
    (which are tie-breakers only), plus any explicitly requested ``extra_objectives``.
    Deterministic order (sorted) so reports are stable.
    """
    cost = set(cost_metrics or ())
    if extra_objectives:
        cost -= set(extra_objectives)
    keys = set()
    for c in candidates:
        keys.update(c.keys())
    return sorted(k for k in keys if k not in cost)


def dominates(a: dict, b: dict, metrics, directions=None) -> bool:
    """Does candidate ``a`` Pareto-dominate candidate ``b`` over ``metrics``?

    ``a`` dominates ``b`` iff over every metric where BOTH have a numeric value ``a`` is
    no worse, AND on at least one such metric ``a`` is strictly better. Missing (``None``
    / non-numeric) values on either side cause that metric to be skipped — see module
    docstring. If the two candidates share no comparable metric, neither dominates.
    """
    strictly_better_somewhere = False
    comparable = False
    for m in metrics:
        av, bv = a.get(m), b.get(m)
        if not (_is_num(av) and _is_num(bv)):
            continue  # missing/unknown on either side -> not comparable on this axis
        comparable = True
        d = _direction(m, directions)
        if _worse(av, bv, d):
            return False  # a is worse somewhere -> cannot dominate
        if _better(av, bv, d):
            strictly_better_somewhere = True
    return comparable and strictly_better_somewhere


def pareto_frontier(candidates, metrics=None, directions=None,
                    cost_metrics=DEFAULT_COST_METRICS, extra_objectives=None,
                    key="label"):
    """Return the non-dominated subset of ``candidates`` (input order preserved).

    ``candidates`` is a list of dicts of ``metric -> value-or-None`` (each may also carry
    a ``key`` field naming the candidate; it is ignored by the math). A candidate is on
    the frontier iff no *other* candidate dominates it under :func:`dominates`.
    """
    if metrics is None:
        metrics = objective_metrics(candidates, cost_metrics=cost_metrics,
                                    extra_objectives=extra_objectives)
    front = []
    for i, c in enumerate(candidates):
        dominated = False
        for j, other in enumerate(candidates):
            if i == j:
                continue
            if dominates(other, c, metrics, directions):
                # tie guard: if they are mutually equal on all comparable metrics,
                # `dominates` is False both ways, so identical rows both survive.
                dominated = True
                break
        if not dominated:
            front.append(c)
    return front


def tie_break(candidates, cost_metrics=DEFAULT_COST_METRICS, directions=None,
              key="label"):
    """Order mutually non-dominated candidates by cost tie-breakers (best first).

    Lower ``vram_gb`` is better, higher ``throughput_pairs_per_sec`` is better. Candidates
    missing a cost value sort *after* those that have it (a known cost beats an unknown
    one), and ties fall back to ``key`` for determinism.
    """
    cost = list(cost_metrics or ())

    def sort_key(c):
        parts = []
        for m in cost:
            v = c.get(m)
            d = _direction(m, directions)
            if _is_num(v):
                # primary: have-a-value (0) before missing (1); secondary: the ordered value
                parts.append((0, v if d == LOWER else -v))
            else:
                parts.append((1, 0.0))
        parts.append(str(c.get(key, "")))
        return tuple(parts)

    return sorted(candidates, key=sort_key)
