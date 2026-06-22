"""Validate the v6.1 dense-retriever experiment config (pure stdlib, no ML).

v6.1 is **DENSE-ONLY**: it improves the Boldt dense RAG embedder's WebFAQ top-50 recall. This config
must never enable reranker training (`reranker_training_enabled` must be ``false``), public benchmarks
stay eval-only, the training mix must sum to 1.0, and every target metric must be numeric. The loader
**fails closed** — an invalid config raises rather than silently proceeding.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List

EXPERIMENT_ID = "v6.1-dense-top50"
MIX_TOLERANCE = 1e-6
REQUIRED_TARGET_METRICS = (
    "webfaq_recall_at_50_min", "webfaq_recall_at_100_min", "webfaq_missing_positive_rate_max",
    "webfaq_ndcg_at_10_min", "germanquad_ndcg_at_10_min", "dt_test_ndcg_at_10_min",
    "matryoshka_256_retention_min",
)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_v6_1_dense_config(cfg: Dict[str, Any]) -> List[str]:
    """Return a list of validation errors (empty == valid). Pure function."""
    if not isinstance(cfg, dict):
        return ["config must be a JSON object"]
    errors: List[str] = []

    if cfg.get("experiment_id") != EXPERIMENT_ID:
        errors.append(f"experiment_id must be {EXPERIMENT_ID!r} (got {cfg.get('experiment_id')!r})")
    for key in ("goal", "base_checkpoint"):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            errors.append(f"{key} must be a non-empty string")

    # DENSE-ONLY: reranker training must be explicitly disabled (no reranker work in v6.1).
    if cfg.get("reranker_training_enabled") is not False:
        errors.append("reranker_training_enabled must be false (v6.1 is dense-only)")

    # public benchmarks are eval-only.
    if cfg.get("public_benchmarks_eval_only") is not True:
        errors.append("public_benchmarks_eval_only must be true")

    # training mix: non-negative numeric fractions summing to 1.0.
    mix = cfg.get("training_mix")
    if not isinstance(mix, dict) or not mix:
        errors.append("training_mix must be a non-empty object")
    else:
        bad = [k for k, v in mix.items() if not _is_number(v) or v < 0]
        if bad:
            errors.append(f"training_mix fractions must be non-negative numbers: {sorted(bad)}")
        else:
            total = sum(mix.values())
            if abs(total - 1.0) > MIX_TOLERANCE:
                errors.append(f"training_mix fractions must sum to 1.0 (got {total})")

    # target metrics: all required keys present and numeric.
    tm = cfg.get("target_metrics")
    if not isinstance(tm, dict):
        errors.append("target_metrics must be an object")
    else:
        for k in REQUIRED_TARGET_METRICS:
            if k not in tm:
                errors.append(f"target_metrics missing '{k}'")
            elif not _is_number(tm[k]):
                errors.append(f"target_metrics['{k}'] must be numeric (got {tm[k]!r})")

    # hard-negative sources: non-empty list of strings.
    hns = cfg.get("hard_negative_sources")
    if not isinstance(hns, list) or not hns or not all(isinstance(x, str) and x for x in hns):
        errors.append("hard_negative_sources must be a non-empty list of strings")

    return errors


def load_v6_1_dense_config(path: Any) -> Dict[str, Any]:
    """Load + validate the v6.1 dense config. Raises ValueError (fail-closed) on any error."""
    cfg = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    errors = validate_v6_1_dense_config(cfg)
    if errors:
        raise ValueError("invalid v6.1 dense config: " + "; ".join(errors))
    return cfg


def is_reranker_training_allowed(cfg: Dict[str, Any]) -> bool:
    """v6.1 never allows reranker training — always False for a valid v6.1 config."""
    return bool(cfg.get("reranker_training_enabled"))
