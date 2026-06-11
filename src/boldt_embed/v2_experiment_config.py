"""v2 data-scale-generalization experiment config (pure stdlib, no ML deps).

Loads/validates `configs/experiments/v2_generalization.json`. Same convention as
:mod:`boldt_embed.config`: ``validate_*`` returns a list of problems (never raises),
``load_*`` raises ``ValueError`` with all problems joined.

Hard rules enforced:
- domain ``target_fraction`` values sum to 1.0 (within tolerance);
- ``public_benchmarks_eval_only`` MUST be true (public test data never trains);
- candidate counts positive with stretch >= min;
- every success criterion is numeric.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

FRACTION_TOLERANCE = 1e-6


def _read(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_pos_int(d: Dict[str, Any], key: str, errors: List[str]) -> None:
    v = d.get(key)
    if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
        errors.append(f"'{key}' must be a positive integer")


@dataclass
class V2ExperimentConfig:
    experiment_id: str
    target_candidate_count_min: int
    target_candidate_count_stretch: int
    public_benchmarks_eval_only: bool
    held_out_eval_sets: List[str]
    student_models: List[str]
    reranker: Dict[str, Any]
    domains: Dict[str, Dict[str, Any]]
    success_criteria: Dict[str, float]
    raw: Dict[str, Any] = field(default_factory=dict)

    def domain_fractions(self) -> Dict[str, float]:
        return {k: float(v["target_fraction"]) for k, v in self.domains.items()}

    def target_counts_by_domain(self, total: int) -> Dict[str, int]:
        """Deterministic integer allocation of `total` candidates across domains by fraction."""
        return {k: int(round(total * float(v["target_fraction"]))) for k, v in self.domains.items()}


def validate_v2_experiment(d: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(d.get("experiment_id"), str) or not d["experiment_id"].strip():
        errors.append("'experiment_id' must be a non-empty string")
    _check_pos_int(d, "target_candidate_count_min", errors)
    _check_pos_int(d, "target_candidate_count_stretch", errors)
    mn, st = d.get("target_candidate_count_min"), d.get("target_candidate_count_stretch")
    if isinstance(mn, int) and isinstance(st, int) and not isinstance(mn, bool) and st < mn:
        errors.append("target_candidate_count_stretch must be >= target_candidate_count_min")

    if d.get("public_benchmarks_eval_only") is not True:
        errors.append("public_benchmarks_eval_only MUST be true (public test data never trains)")

    if not isinstance(d.get("held_out_eval_sets"), list) or not d["held_out_eval_sets"]:
        errors.append("'held_out_eval_sets' must be a non-empty list")
    if not isinstance(d.get("student_models"), list) or not d["student_models"]:
        errors.append("'student_models' must be a non-empty list")

    rr = d.get("reranker")
    if not isinstance(rr, dict):
        errors.append("'reranker' must be an object")
    else:
        if not isinstance(rr.get("train"), bool):
            errors.append("reranker.train must be a bool")
        if not isinstance(rr.get("loss"), str) or not rr["loss"].strip():
            errors.append("reranker.loss must be a non-empty string")
        if not isinstance(rr.get("candidate_sources"), list) or not rr["candidate_sources"]:
            errors.append("reranker.candidate_sources must be a non-empty list")
        npq = rr.get("negatives_per_query")
        if not isinstance(npq, int) or isinstance(npq, bool) or npq <= 0:
            errors.append("reranker.negatives_per_query must be a positive integer")

    domains = d.get("domains")
    if not isinstance(domains, dict) or not domains:
        errors.append("'domains' must be a non-empty object")
    else:
        total = 0.0
        for name, spec in domains.items():
            if not isinstance(spec, dict) or not _is_number(spec.get("target_fraction")):
                errors.append(f"domain '{name}' must have a numeric target_fraction")
            else:
                if not (0.0 <= float(spec["target_fraction"]) <= 1.0):
                    errors.append(f"domain '{name}' target_fraction must be in [0, 1]")
                total += float(spec["target_fraction"])
        if abs(total - 1.0) > FRACTION_TOLERANCE:
            errors.append(f"domain target_fraction values must sum to 1.0 (got {total:.6f})")

    sc = d.get("success_criteria")
    if not isinstance(sc, dict) or not sc:
        errors.append("'success_criteria' must be a non-empty object")
    else:
        for k, v in sc.items():
            if not _is_number(v):
                errors.append(f"success_criteria.{k} must be numeric")
    return errors


def load_v2_experiment_config(path: str | Path) -> V2ExperimentConfig:
    d = _read(path)
    errors = validate_v2_experiment(d)
    if errors:
        raise ValueError("invalid v2 experiment config: " + "; ".join(errors))
    return V2ExperimentConfig(
        experiment_id=d["experiment_id"],
        target_candidate_count_min=int(d["target_candidate_count_min"]),
        target_candidate_count_stretch=int(d["target_candidate_count_stretch"]),
        public_benchmarks_eval_only=bool(d["public_benchmarks_eval_only"]),
        held_out_eval_sets=list(d["held_out_eval_sets"]),
        student_models=list(d["student_models"]),
        reranker=dict(d["reranker"]),
        domains=dict(d["domains"]),
        success_criteria=dict(d["success_criteria"]),
        raw=d,
    )
