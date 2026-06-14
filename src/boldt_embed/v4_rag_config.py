"""v4 RAG-reranker experiment config (pure stdlib, no ML deps).

Loads/validates `configs/experiments/v4_rag_reranker.json`. Same convention as
:mod:`boldt_embed.v3_experiment_config`: ``validate_*`` returns a list of problems (never
raises), ``load_*`` raises ``ValueError`` with all problems joined.

v4 retargets the product goal from "legal/admin domain transfer" to "a good German RAG
reranker". Legal eval (GerDaLIR) is kept as a **diagnostic only** — it is no longer a release
blocker. Hard rules:

- ``legal_eval_is_diagnostic_only`` MUST be true (legal is diagnostic, not a gate);
- ``public_benchmarks_eval_only`` MUST be true (public test data never trains);
- ``candidate_sources`` is non-empty;
- ``train_domains`` MUST NOT include any public-benchmark / eval source;
- every ``success_criteria`` value is numeric.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

# Public-benchmark / eval-only sources that may NEVER appear as a training domain.
PUBLIC_BENCHMARK_EVAL = frozenset({
    "germanquad", "dt_test", "gerdalir", "webfaq_heldout", "local_rag",
    "miracl", "miracl_de", "mldr", "mldr_de", "sts22", "sts22_de", "qa_wiki", "legal",
})


def _read(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


@dataclass
class V4RagConfig:
    experiment_id: str
    goal: str
    legal_eval_is_diagnostic_only: bool
    public_benchmarks_eval_only: bool
    dense_default: str
    teacher_reranker: str
    candidate_sources: List[str]
    train_domains: List[str]
    eval_sets: List[str]
    success_criteria: Dict[str, float]
    raw: Dict[str, Any] = field(default_factory=dict)


def validate_v4_rag(d: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(d.get("experiment_id"), str) or not d["experiment_id"].strip():
        errors.append("'experiment_id' must be a non-empty string")
    if not isinstance(d.get("goal"), str) or not d["goal"].strip():
        errors.append("'goal' must be a non-empty string")

    if d.get("legal_eval_is_diagnostic_only") is not True:
        errors.append("legal_eval_is_diagnostic_only MUST be true (legal is diagnostic, not a gate)")
    if d.get("public_benchmarks_eval_only") is not True:
        errors.append("public_benchmarks_eval_only MUST be true (public test data never trains)")

    for key in ("dense_default", "teacher_reranker"):
        if not isinstance(d.get(key), str) or not d[key].strip():
            errors.append(f"'{key}' must be a non-empty string")

    cs = d.get("candidate_sources")
    if not isinstance(cs, list) or not cs or not all(isinstance(s, str) and s.strip() for s in cs):
        errors.append("'candidate_sources' must be a non-empty list of strings")

    td = d.get("train_domains")
    if not isinstance(td, list) or not td or not all(isinstance(s, str) and s.strip() for s in td):
        errors.append("'train_domains' must be a non-empty list of strings")
    else:
        leaked = sorted(set(td) & PUBLIC_BENCHMARK_EVAL)
        if leaked:
            errors.append(f"train_domains must not include public-benchmark/eval sources: {leaked}")

    if not isinstance(d.get("eval_sets"), list) or not d["eval_sets"]:
        errors.append("'eval_sets' must be a non-empty list")

    sc = d.get("success_criteria")
    if not isinstance(sc, dict) or not sc:
        errors.append("'success_criteria' must be a non-empty object")
    else:
        for k, v in sc.items():
            if not _is_number(v):
                errors.append(f"success_criteria.{k} must be numeric")
    return errors


def load_v4_rag_config(path: str | Path) -> V4RagConfig:
    d = _read(path)
    errors = validate_v4_rag(d)
    if errors:
        raise ValueError("invalid v4 RAG config: " + "; ".join(errors))
    return V4RagConfig(
        experiment_id=d["experiment_id"], goal=d["goal"],
        legal_eval_is_diagnostic_only=bool(d["legal_eval_is_diagnostic_only"]),
        public_benchmarks_eval_only=bool(d["public_benchmarks_eval_only"]),
        dense_default=d["dense_default"], teacher_reranker=d["teacher_reranker"],
        candidate_sources=list(d["candidate_sources"]), train_domains=list(d["train_domains"]),
        eval_sets=list(d["eval_sets"]), success_criteria=dict(d["success_criteria"]), raw=d)
