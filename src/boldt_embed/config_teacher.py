"""Teacher/student-2026 config dataclasses + validation (pure stdlib, no ML deps).

This is the configuration layer for the 2026 teacher→student distillation workflow:

* ``teacher_models.json``        -> :func:`load_teacher_models_config`
* ``student_training_2026.json`` -> :func:`load_student_training_config`

Mirrors the conventions in :mod:`boldt_embed.config`: each ``validate_*`` returns a list
of human-readable problems (never raises), and each ``load_*`` raises ``ValueError`` with
all problems joined. Loaders never import torch / sentence-transformers — they only read
and validate JSON, so they run inside the stdlib unit-test gate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reuse the shared primitives so teacher/student validation stays consistent with the
# rest of the config layer (same messages for matryoshka / positive-int / non-empty-str).
from .config import _check_matryoshka, _check_pos_int, _check_str  # noqa: F401

EMBEDDING_BACKENDS = {"sentence_transformers", "transformers_custom"}
RERANKER_BACKENDS = {"sentence_transformers_cross_encoder", "transformers_custom"}
DTYPES = {"bfloat16", "float16", "float32"}
SCORE_ACTIVATIONS = {"raw", "sigmoid"}
STUDENT_VARIANTS = {"causal", "bidirectional"}
# Only one split policy is permitted: public benchmark *test* data must stay eval-only.
# Keeping this an allow-list makes the anti-leakage rule a hard config error, not a comment.
SPLIT_POLICIES = {"public_benchmarks_eval_only"}


def _read(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _check_dtype(d: Dict[str, Any], errors: List[str], prefix: str) -> None:
    dt = d.get("torch_dtype")
    if dt is not None and dt not in DTYPES:
        errors.append(f"{prefix}torch_dtype '{dt}' not in {sorted(DTYPES)}")


# --------------------------------------------------------------------- teacher models
@dataclass
class EmbeddingTeacherConfig:
    model_name: str
    backend: str
    device: str
    torch_dtype: str
    max_length: int
    batch_size: int
    query_instruction: str
    document_instruction: Optional[str]
    output_dim: Optional[int]
    normalize: bool
    use_flash_attention_2_if_available: bool
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankerTeacherConfig:
    model_name: str
    backend: str
    device: str
    torch_dtype: str
    max_length: int
    batch_size: int
    instruction: str
    score_activation: str
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TeacherModelsConfig:
    embedding_teacher: EmbeddingTeacherConfig
    reranker_teacher: RerankerTeacherConfig
    raw: Dict[str, Any] = field(default_factory=dict)


def validate_embedding_teacher(d: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(d, dict):
        return ["'embedding_teacher' must be an object"]
    _check_str(d, "model_name", errors)
    _check_str(d, "backend", errors)
    if isinstance(d.get("backend"), str) and d["backend"] not in EMBEDDING_BACKENDS:
        errors.append(f"embedding_teacher.backend '{d['backend']}' not in {sorted(EMBEDDING_BACKENDS)}")
    _check_pos_int(d, "max_length", errors)
    _check_pos_int(d, "batch_size", errors)
    _check_pos_int(d, "output_dim", errors, required=False)
    _check_dtype(d, errors, "embedding_teacher.")
    return errors


def validate_reranker_teacher(d: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(d, dict):
        return ["'reranker_teacher' must be an object"]
    _check_str(d, "model_name", errors)
    _check_str(d, "backend", errors)
    if isinstance(d.get("backend"), str) and d["backend"] not in RERANKER_BACKENDS:
        errors.append(f"reranker_teacher.backend '{d['backend']}' not in {sorted(RERANKER_BACKENDS)}")
    _check_str(d, "instruction", errors)
    _check_pos_int(d, "max_length", errors)
    _check_pos_int(d, "batch_size", errors)
    act = d.get("score_activation", "raw")
    if act not in SCORE_ACTIVATIONS:
        errors.append(f"reranker_teacher.score_activation '{act}' not in {sorted(SCORE_ACTIVATIONS)}")
    _check_dtype(d, errors, "reranker_teacher.")
    return errors


def validate_teacher_models(d: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if "embedding_teacher" not in d:
        errors.append("missing required section 'embedding_teacher'")
    else:
        errors += validate_embedding_teacher(d["embedding_teacher"])
    if "reranker_teacher" not in d:
        errors.append("missing required section 'reranker_teacher'")
    else:
        errors += validate_reranker_teacher(d["reranker_teacher"])
    return errors


def _embedding_teacher(d: Dict[str, Any]) -> EmbeddingTeacherConfig:
    return EmbeddingTeacherConfig(
        model_name=d["model_name"],
        backend=d["backend"],
        device=d.get("device", "cuda"),
        torch_dtype=d.get("torch_dtype", "bfloat16"),
        max_length=int(d["max_length"]),
        batch_size=int(d["batch_size"]),
        query_instruction=d.get("query_instruction", ""),
        document_instruction=d.get("document_instruction"),
        output_dim=(int(d["output_dim"]) if d.get("output_dim") is not None else None),
        normalize=bool(d.get("normalize", True)),
        use_flash_attention_2_if_available=bool(d.get("use_flash_attention_2_if_available", True)),
        raw=d,
    )


def _reranker_teacher(d: Dict[str, Any]) -> RerankerTeacherConfig:
    return RerankerTeacherConfig(
        model_name=d["model_name"],
        backend=d["backend"],
        device=d.get("device", "cuda"),
        torch_dtype=d.get("torch_dtype", "bfloat16"),
        max_length=int(d["max_length"]),
        batch_size=int(d["batch_size"]),
        instruction=d["instruction"],
        score_activation=d.get("score_activation", "raw"),
        raw=d,
    )


def load_teacher_models_config(path: str | Path) -> TeacherModelsConfig:
    d = _read(path)
    errors = validate_teacher_models(d)
    if errors:
        raise ValueError("invalid teacher_models config: " + "; ".join(errors))
    return TeacherModelsConfig(
        embedding_teacher=_embedding_teacher(d["embedding_teacher"]),
        reranker_teacher=_reranker_teacher(d["reranker_teacher"]),
        raw=d,
    )


# ------------------------------------------------------------------- student training
@dataclass
class StudentTrainingConfig:
    base_model: str
    student_variant: str
    matryoshka_dims: List[int]
    target_dim: int
    losses: List[str]
    train_eval_split_policy: str
    hardware_profile: str
    raw: Dict[str, Any] = field(default_factory=dict)


def validate_student_training(d: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    _check_str(d, "base_model", errors)
    variant = d.get("student_variant")
    if variant not in STUDENT_VARIANTS:
        errors.append(f"student_variant '{variant}' not in {sorted(STUDENT_VARIANTS)}")
    _check_matryoshka(d.get("matryoshka_dims"), d.get("target_dim"), errors)
    _check_pos_int(d, "target_dim", errors)
    dims = d.get("matryoshka_dims")
    if isinstance(dims, list) and dims and isinstance(d.get("target_dim"), int):
        if d["target_dim"] not in dims:
            errors.append(f"target_dim {d['target_dim']} must be one of matryoshka_dims {dims}")
    losses = d.get("losses")
    if not isinstance(losses, list) or not losses or not all(isinstance(x, str) and x.strip() for x in losses):
        errors.append("'losses' must be a non-empty list of non-empty strings")
    policy = d.get("train_eval_split_policy")
    if policy not in SPLIT_POLICIES:
        errors.append(
            f"train_eval_split_policy '{policy}' not in {sorted(SPLIT_POLICIES)} "
            "(public benchmark test data must stay eval-only)"
        )
    _check_str(d, "hardware_profile", errors)
    return errors


def load_student_training_config(path: str | Path) -> StudentTrainingConfig:
    d = _read(path)
    errors = validate_student_training(d)
    if errors:
        raise ValueError("invalid student_training config: " + "; ".join(errors))
    return StudentTrainingConfig(
        base_model=d["base_model"],
        student_variant=d["student_variant"],
        matryoshka_dims=list(d["matryoshka_dims"]),
        target_dim=int(d["target_dim"]),
        losses=list(d["losses"]),
        train_eval_split_policy=d["train_eval_split_policy"],
        hardware_profile=d["hardware_profile"],
        raw=d,
    )
