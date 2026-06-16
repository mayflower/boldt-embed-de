"""Rerank-policy artifact loader/validator (pure stdlib, no ML).

A bounded rerank policy (e.g. `configs/policies/bounded_margin_override_v1.json`) is a versioned
DEPLOYMENT artifact, not a model. It pins the checkpoint + the inference-time bounds and — crucially
— makes accidental raw-always-rerank promotion impossible: validation FAILS if the policy claims
raw always-rerank is recommended, or if a forbidden inference feature (qrels/labels/oracle/…) leaks
into the allowed set. Same convention as the other configs: ``validate_*`` returns a list of
problems (never raises); ``load_*`` raises ``ValueError`` with all problems joined.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REQUIRED_VALIDATION_THRESHOLDS = ("max_germanquad_catastrophic", "min_webfaq_delta_ndcg10",
                                  "min_dt_test_delta_ndcg10")


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def validate_policy(d: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for k in ("policy_id", "policy_type", "recommended_mode"):
        if not _nonempty_str(d.get(k)):
            errors.append(f"'{k}' must be a non-empty string")

    # HARD GUARD: never recommend raw always-rerank.
    if d.get("raw_always_rerank_recommended") is not False:
        errors.append("raw_always_rerank_recommended must be false "
                      "(raw always-rerank is unsafe on near-ceiling lists)")
    if d.get("recommended_mode") not in (None, "policy_gated_only"):
        # only the policy-gated mode is allowed to be recommended
        if d.get("recommended_mode") in ("raw", "always_rerank", "raw_always_rerank"):
            errors.append("recommended_mode must not be a raw always-rerank mode")

    if not _nonempty_str(d.get("model_checkpoint")):
        errors.append("'model_checkpoint' must be a non-empty string (the pinned checkpoint)")

    allowed = d.get("features_allowed_at_inference")
    forbidden = d.get("features_forbidden_at_inference")
    if not isinstance(allowed, list) or not allowed:
        errors.append("'features_allowed_at_inference' must be a non-empty list")
    if not isinstance(forbidden, list) or not forbidden:
        errors.append("'features_forbidden_at_inference' must be a non-empty list")
    if isinstance(allowed, list) and isinstance(forbidden, list):
        overlap = sorted(set(allowed) & set(forbidden))
        if overlap:
            errors.append(f"forbidden inference features overlap allowed features: {overlap}")

    bounds = d.get("bounds")
    if not isinstance(bounds, dict) or not bounds:
        errors.append("policy has no 'bounds' (the inference bounds object is required)")

    val = d.get("validation")
    if not isinstance(val, dict) or not val:
        errors.append("'validation' must be a non-empty object")
    else:
        for t in REQUIRED_VALIDATION_THRESHOLDS:
            if not _is_number(val.get(t)):
                errors.append(f"validation threshold '{t}' missing or non-numeric")

    appl = d.get("applicability")
    if isinstance(appl, dict) and not _nonempty_str(appl.get("task")):
        errors.append("applicability.task must be a non-empty string")
    return errors


def load_policy(path: str | Path) -> Dict[str, Any]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_policy(d)
    if errors:
        raise ValueError("invalid rerank policy: " + "; ".join(errors))
    return d


def check_model_exists(d: Dict[str, Any], root: str | Path = ".",
                       require: bool = False) -> Tuple[bool, Optional[str]]:
    """Check the pinned checkpoint exists on disk. Returns (ok, message). When ``require`` is False a
    missing path is a WARNING (ok=True); when True it is a failure (ok=False)."""
    ckpt = d.get("model_checkpoint")
    if not _nonempty_str(ckpt):
        return (not require, "model_checkpoint not set")
    p = Path(root) / ckpt
    if p.exists():
        return True, f"model checkpoint present: {ckpt}"
    msg = f"model checkpoint NOT found: {ckpt}"
    return (False, msg) if require else (True, "WARNING: " + msg)
