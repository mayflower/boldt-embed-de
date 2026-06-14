"""v3 real-domain-generalization experiment config (pure stdlib, no ML deps).

Loads/validates `configs/experiments/v3_real_domain_generalization.json`. Same convention as
:mod:`boldt_embed.v2_experiment_config`: ``validate_*`` returns a list of problems (never
raises), ``load_*`` raises ``ValueError`` with all problems joined.

v3 encodes the v2 lessons as HARD gates (not advice):
- ``public_benchmarks_eval_only`` MUST be true — public test data never trains;
- ``train_only_if_license_known`` MUST be true — no row trains without a concrete license
  (the v2 teacher cache reported by_license {"unknown": 44336}; that must never reach release);
- ``train_only_if_leakage_full_scan_complete`` MUST be true — the v2 O(n*m) shortcut (leakage
  filtered vs GermanQuAD+DT only, mining over a ~3.5k subset) is not allowed at v3 scale;
- ``license_unknown_rows_max`` MUST be 0 — zero unknown-license rows in training.

Also enforced:
- ``domain_targets`` fractions sum to 1.0 (within tolerance), each in [0, 1];
- candidate / teacher-validated counts are positive ints, validated <= candidate min;
- every success criterion is numeric.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

FRACTION_TOLERANCE = 1e-6

# Boolean flags that MUST be true for a v3 config to be valid (the v2 lessons, as gates).
REQUIRED_TRUE_FLAGS = (
    "public_benchmarks_eval_only",
    "train_only_if_license_known",
    "train_only_if_leakage_full_scan_complete",
)


def _read(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_pos_int(d: Dict[str, Any], key: str, errors: List[str]) -> None:
    v = d.get(key)
    if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
        errors.append(f"'{key}' must be a positive integer")


@dataclass
class V3ExperimentConfig:
    experiment_id: str
    goal: str
    public_benchmarks_eval_only: bool
    train_only_if_license_known: bool
    train_only_if_leakage_full_scan_complete: bool
    target_candidate_count_min: int
    target_teacher_validated_positives_min: int
    domain_targets: Dict[str, float]
    success_criteria: Dict[str, float]
    raw: Dict[str, Any] = field(default_factory=dict)

    def domain_fractions(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.domain_targets.items()}

    def target_counts_by_domain(self, total: int) -> Dict[str, int]:
        """Deterministic integer allocation of `total` candidates across domains by fraction."""
        return {k: int(round(total * float(v))) for k, v in self.domain_targets.items()}


def validate_v3_experiment(d: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(d.get("experiment_id"), str) or not d["experiment_id"].strip():
        errors.append("'experiment_id' must be a non-empty string")
    if not isinstance(d.get("goal"), str) or not d["goal"].strip():
        errors.append("'goal' must be a non-empty string")

    # v2-lesson gates: each flag must be present AND exactly True.
    for flag in REQUIRED_TRUE_FLAGS:
        if d.get(flag) is not True:
            errors.append(f"{flag} MUST be true")

    _check_pos_int(d, "target_candidate_count_min", errors)
    _check_pos_int(d, "target_teacher_validated_positives_min", errors)
    cmin = d.get("target_candidate_count_min")
    pmin = d.get("target_teacher_validated_positives_min")
    if (isinstance(cmin, int) and not isinstance(cmin, bool)
            and isinstance(pmin, int) and not isinstance(pmin, bool) and pmin > cmin):
        errors.append("target_teacher_validated_positives_min must be <= target_candidate_count_min")

    domains = d.get("domain_targets")
    if not isinstance(domains, dict) or not domains:
        errors.append("'domain_targets' must be a non-empty object")
    else:
        total = 0.0
        for name, frac in domains.items():
            if not _is_number(frac):
                errors.append(f"domain '{name}' target fraction must be numeric")
            else:
                if not (0.0 <= float(frac) <= 1.0):
                    errors.append(f"domain '{name}' target fraction must be in [0, 1]")
                total += float(frac)
        if abs(total - 1.0) > FRACTION_TOLERANCE:
            errors.append(f"domain target fractions must sum to 1.0 (got {total:.6f})")

    sc = d.get("success_criteria")
    if not isinstance(sc, dict) or not sc:
        errors.append("'success_criteria' must be a non-empty object")
    else:
        for k, v in sc.items():
            if not _is_number(v):
                errors.append(f"success_criteria.{k} must be numeric")
        # The license gate: zero unknown-license rows allowed in training.
        if "license_unknown_rows_max" not in sc:
            errors.append("success_criteria.license_unknown_rows_max is required")
        elif _is_number(sc["license_unknown_rows_max"]) and sc["license_unknown_rows_max"] != 0:
            errors.append("success_criteria.license_unknown_rows_max MUST be 0")
    return errors


def load_v3_experiment_config(path: str | Path) -> V3ExperimentConfig:
    d = _read(path)
    errors = validate_v3_experiment(d)
    if errors:
        raise ValueError("invalid v3 experiment config: " + "; ".join(errors))
    return V3ExperimentConfig(
        experiment_id=d["experiment_id"],
        goal=d["goal"],
        public_benchmarks_eval_only=bool(d["public_benchmarks_eval_only"]),
        train_only_if_license_known=bool(d["train_only_if_license_known"]),
        train_only_if_leakage_full_scan_complete=bool(d["train_only_if_leakage_full_scan_complete"]),
        target_candidate_count_min=int(d["target_candidate_count_min"]),
        target_teacher_validated_positives_min=int(d["target_teacher_validated_positives_min"]),
        domain_targets=dict(d["domain_targets"]),
        success_criteria=dict(d["success_criteria"]),
        raw=d,
    )
