"""Config dataclasses + validation for the three tracks and evaluation (pure stdlib).

Loaders raise ``ValueError`` on an invalid config; ``validate_config_dict`` returns the
list of problems without raising (used by the smoke suite and dry-run trainers).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

POOLINGS = {
    "mean", "eos", "last_token", "eos_or_last_token", "cls",
    "latent_attention", "latent_attention_optional",
}


def _read(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _check_matryoshka(dims: Any, embedding_dim: Any, errors: List[str], prefix: str = "") -> None:
    if not isinstance(dims, list) or not dims or not all(isinstance(x, int) and x > 0 for x in dims):
        errors.append(f"{prefix}matryoshka_dims must be a non-empty list of positive ints")
        return
    if dims != sorted(dims, reverse=True) or len(set(dims)) != len(dims):
        errors.append(f"{prefix}matryoshka_dims must be strictly decreasing")
    if isinstance(embedding_dim, int) and max(dims) > embedding_dim:
        errors.append(
            f"{prefix}largest matryoshka dim {max(dims)} exceeds embedding_dim {embedding_dim}"
        )


def _check_pos_int(d: Dict[str, Any], key: str, errors: List[str], required: bool = True) -> None:
    if key not in d:
        if required:
            errors.append(f"missing required field '{key}'")
        return
    if not isinstance(d[key], int) or isinstance(d[key], bool) or d[key] <= 0:
        errors.append(f"'{key}' must be a positive integer")


def _check_str(d: Dict[str, Any], key: str, errors: List[str], required: bool = True) -> None:
    if key not in d:
        if required:
            errors.append(f"missing required field '{key}'")
        return
    if not isinstance(d[key], str) or not d[key].strip():
        errors.append(f"'{key}' must be a non-empty string")


# --------------------------------------------------------------------------- causal
@dataclass
class CausalConfig:
    model_name_or_path: str
    variant: str
    pooling: str
    normalize_embeddings: bool
    embedding_dim: int
    matryoshka_dims: List[int]
    query_instruction: str
    document_instruction: str
    loss: str
    temperature: float
    max_query_length: int
    max_document_length: int
    dtype: str
    dry_run: bool
    raw: Dict[str, Any] = field(default_factory=dict)


def validate_causal(d: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    _check_str(d, "model_name_or_path", errors)
    _check_str(d, "pooling", errors)
    if isinstance(d.get("pooling"), str) and d["pooling"] not in POOLINGS:
        errors.append(f"unknown pooling '{d['pooling']}' (allowed: {sorted(POOLINGS)})")
    _check_pos_int(d, "embedding_dim", errors)
    _check_matryoshka(d.get("matryoshka_dims"), d.get("embedding_dim"), errors)
    temp = d.get("temperature", 0.05)
    if not isinstance(temp, (int, float)) or isinstance(temp, bool) or not (0.0 < temp <= 1.0):
        errors.append("'temperature' must be a float in (0, 1]")
    _check_pos_int(d, "max_query_length", errors, required=False)
    _check_pos_int(d, "max_document_length", errors, required=False)
    if "{query}" not in d.get("query_instruction", "{query}"):
        errors.append("'query_instruction' must contain the '{query}' placeholder")
    return errors


def load_causal_config(path: str | Path) -> CausalConfig:
    d = _read(path)
    errors = validate_causal(d)
    if errors:
        raise ValueError("invalid causal config: " + "; ".join(errors))
    return CausalConfig(
        model_name_or_path=d["model_name_or_path"],
        variant=d.get("variant", "causal"),
        pooling=d["pooling"],
        normalize_embeddings=bool(d.get("normalize_embeddings", True)),
        embedding_dim=int(d["embedding_dim"]),
        matryoshka_dims=list(d["matryoshka_dims"]),
        query_instruction=d.get("query_instruction", "{query}"),
        document_instruction=d.get("document_instruction", "{document}"),
        loss=d.get("loss", "multiple_negatives_ranking_loss"),
        temperature=float(d.get("temperature", 0.05)),
        max_query_length=int(d.get("max_query_length", 256)),
        max_document_length=int(d.get("max_document_length", 512)),
        dtype=d.get("dtype", "bfloat16"),
        dry_run=bool(d.get("dry_run", False)),
        raw=d,
    )


# -------------------------------------------------------------------- bidirectional
@dataclass
class BidirectionalConfig:
    model_name_or_path: str
    variant: str
    adaptation: str
    pooling_ablation: List[str]
    normalize_embeddings: bool
    embedding_dim: int
    matryoshka_dims: List[int]
    mntp_steps_dry_run: int
    contrastive_steps_dry_run: int
    checkpoint_merging: List[str]
    dry_run: bool
    raw: Dict[str, Any] = field(default_factory=dict)


def validate_bidirectional(d: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    _check_str(d, "model_name_or_path", errors)
    _check_str(d, "adaptation", errors)
    _check_pos_int(d, "embedding_dim", errors)
    _check_matryoshka(d.get("matryoshka_dims"), d.get("embedding_dim"), errors)
    ablation = d.get("pooling_ablation", [])
    if not isinstance(ablation, list) or not ablation:
        errors.append("'pooling_ablation' must be a non-empty list")
    else:
        for p in ablation:
            if p not in POOLINGS:
                errors.append(f"unknown pooling in ablation: '{p}'")
    return errors


def load_bidirectional_config(path: str | Path) -> BidirectionalConfig:
    d = _read(path)
    errors = validate_bidirectional(d)
    if errors:
        raise ValueError("invalid bidirectional config: " + "; ".join(errors))
    return BidirectionalConfig(
        model_name_or_path=d["model_name_or_path"],
        variant=d.get("variant", "bidirectional"),
        adaptation=d["adaptation"],
        pooling_ablation=list(d["pooling_ablation"]),
        normalize_embeddings=bool(d.get("normalize_embeddings", True)),
        embedding_dim=int(d["embedding_dim"]),
        matryoshka_dims=list(d["matryoshka_dims"]),
        mntp_steps_dry_run=int(d.get("mntp_steps_dry_run", d.get("mmtp_steps_dry_run", 0))),
        contrastive_steps_dry_run=int(d.get("contrastive_steps_dry_run", 0)),
        checkpoint_merging=list(d.get("checkpoint_merging", [])),
        dry_run=bool(d.get("dry_run", False)),
        raw=d,
    )


# ------------------------------------------------------------------------- reranker
@dataclass
class RerankerConfig:
    model_name_or_path: str
    variant: str
    input_template: str
    output_mode: str
    positive_label: str
    negative_label: str
    max_length: int
    hard_negative_sources: List[str]
    dry_run: bool
    raw: Dict[str, Any] = field(default_factory=dict)


def validate_reranker(d: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    _check_str(d, "model_name_or_path", errors)
    _check_str(d, "input_template", errors)
    tmpl = d.get("input_template", "")
    if "{query}" not in tmpl or "{document}" not in tmpl:
        errors.append("'input_template' must contain '{query}' and '{document}'")
    _check_str(d, "positive_label", errors)
    _check_str(d, "negative_label", errors)
    _check_pos_int(d, "max_length", errors)
    return errors


def load_reranker_config(path: str | Path) -> RerankerConfig:
    d = _read(path)
    errors = validate_reranker(d)
    if errors:
        raise ValueError("invalid reranker config: " + "; ".join(errors))
    return RerankerConfig(
        model_name_or_path=d["model_name_or_path"],
        variant=d.get("variant", "reranker"),
        input_template=d["input_template"],
        output_mode=d.get("output_mode", "binary_logit_or_scalar_score"),
        positive_label=d.get("positive_label", "Ja"),
        negative_label=d.get("negative_label", "Nein"),
        max_length=int(d["max_length"]),
        hard_negative_sources=list(d.get("hard_negative_sources", [])),
        dry_run=bool(d.get("dry_run", False)),
        raw=d,
    )


# ----------------------------------------------------------------------- evaluation
@dataclass
class EvaluationConfig:
    metrics: List[str]
    matryoshka_dims: List[int]
    stress_tests: List[str]
    report_metadata_required: List[str]
    raw: Dict[str, Any] = field(default_factory=dict)


def validate_evaluation(d: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(d.get("metrics"), list) or not d["metrics"]:
        errors.append("'metrics' must be a non-empty list")
    _check_matryoshka(d.get("matryoshka_dims"), None, errors)
    if not isinstance(d.get("report_metadata_required"), list) or not d["report_metadata_required"]:
        errors.append("'report_metadata_required' must be a non-empty list")
    return errors


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    d = _read(path)
    errors = validate_evaluation(d)
    if errors:
        raise ValueError("invalid evaluation config: " + "; ".join(errors))
    return EvaluationConfig(
        metrics=list(d["metrics"]),
        matryoshka_dims=list(d.get("matryoshka_dims", [])),
        stress_tests=list(d.get("stress_tests", [])),
        report_metadata_required=list(d["report_metadata_required"]),
        raw=d,
    )


def validate_config_dict(d: Dict[str, Any]) -> List[str]:
    """Dispatch validation by config shape; returns a list of problems (never raises)."""
    if "metrics" in d:
        return validate_evaluation(d)
    variant = d.get("variant")
    if variant == "causal":
        return validate_causal(d)
    if variant == "bidirectional":
        return validate_bidirectional(d)
    if variant == "reranker":
        return validate_reranker(d)
    return [f"cannot classify config (variant={variant!r}, no 'metrics' key)"]
