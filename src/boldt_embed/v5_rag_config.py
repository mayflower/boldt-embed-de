"""v5 small-German-RAG experiment config (pure stdlib, no ML deps).

Loads/validates `configs/experiments/v5_small_rag.json`. Same convention as
:mod:`boldt_embed.v4_rag_config`: ``validate_*`` returns a list of problems (never raises),
``load_*`` raises ``ValueError`` with all problems joined.

v5 retargets from "a good FAQ reranker" (v4, which lifted WebFAQ +0.29 but degraded GermanQuAD
−0.07 and did not generalize) to a **small, deployable German RAG retriever + reranker** trained
on diverse question styles. Legal eval (GerDaLIR) stays **diagnostic only**. Two design lessons
from v4 are encoded as hard rules:

- public-benchmark eval sets (GermanQuAD/DT-test/WebFAQ/…) may NEVER appear as a training domain;
- **near-ceiling** eval sets (oracle nDCG@10 ≥ threshold) must not be a *primary* promotion
  signal and only carry a small do-not-regress tolerance — reranking a near-perfect list can
  only churn it, so a tiny negative delta there is noise, not failure.

Hard rules enforced here:

- ``legal_eval_is_diagnostic_only`` MUST be true;
- ``public_benchmarks_eval_only`` MUST be true;
- ``dense_candidates`` and ``reranker_candidates`` are each non-empty lists of strings;
- ``teacher_models`` provides non-empty ``embedding`` and ``reranker``;
- no ``train_domains`` entry is (or embeds) a public-benchmark eval source;
- ``near_ceiling_eval_policy.use_do_not_regress_tolerance`` is in [-0.02, 0];
- every ``success_criteria`` value is numeric.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

# Public-benchmark tokens: any eval/train name containing one is a public set that must stay
# eval-only and may never train. Private/local sets (local_rag, *_private) are intentionally
# allowed to appear in both train_domains and eval_sets (trained on a disjoint split).
# Public-benchmark EVAL identifiers that must stay eval-only. Note: the WebFAQ token is
# "webfaq_heldout" (the held-out eval split), NOT a blanket "webfaq" — WebFAQ FAQ training pairs
# (faq_real) and WebFAQ 2.0 hard negatives (webfaq2) are legitimate TRAINING sources and must not
# be flagged. germandpr/germanquad/dt_test/gerdalir are public QA/IR benchmarks (eval-only).
PUBLIC_BENCHMARK_TOKENS = frozenset({
    "germanquad", "dt_test", "gerdalir", "webfaq_heldout", "germandpr",
    "miracl", "mldr", "sts22", "mmteb", "mteb",
})


def _read(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def is_public_benchmark(name: str) -> bool:
    """True if ``name`` references a public benchmark (so it must stay eval-only)."""
    n = name.lower()
    return any(tok in n for tok in PUBLIC_BENCHMARK_TOKENS)


def _nonempty_str_list(x: Any) -> bool:
    return isinstance(x, list) and bool(x) and all(isinstance(s, str) and s.strip() for s in x)


@dataclass
class V5RagConfig:
    experiment_id: str
    goal: str
    legal_eval_is_diagnostic_only: bool
    public_benchmarks_eval_only: bool
    dense_candidates: List[str]
    reranker_candidates: List[str]
    teacher_models: Dict[str, str]
    train_domains: List[str]
    eval_sets: List[str]
    success_criteria: Dict[str, float]
    near_ceiling_eval_policy: Dict[str, Any]
    raw: Dict[str, Any] = field(default_factory=dict)


def validate_v5_rag(d: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    if not isinstance(d.get("experiment_id"), str) or not d["experiment_id"].strip():
        errors.append("'experiment_id' must be a non-empty string")
    if not isinstance(d.get("goal"), str) or not d["goal"].strip():
        errors.append("'goal' must be a non-empty string")

    if d.get("legal_eval_is_diagnostic_only") is not True:
        errors.append("legal_eval_is_diagnostic_only MUST be true (legal/GerDaLIR is diagnostic, not a gate)")
    if d.get("public_benchmarks_eval_only") is not True:
        errors.append("public_benchmarks_eval_only MUST be true (public test data never trains)")

    if not _nonempty_str_list(d.get("dense_candidates")):
        errors.append("'dense_candidates' must be a non-empty list of strings")
    if not _nonempty_str_list(d.get("reranker_candidates")):
        errors.append("'reranker_candidates' must be a non-empty list of strings")

    tm = d.get("teacher_models")
    if not isinstance(tm, dict):
        errors.append("'teacher_models' must be an object with 'embedding' and 'reranker'")
    else:
        for k in ("embedding", "reranker"):
            if not isinstance(tm.get(k), str) or not tm[k].strip():
                errors.append(f"teacher_models.{k} must be a non-empty string")

    td = d.get("train_domains")
    if not _nonempty_str_list(td):
        errors.append("'train_domains' must be a non-empty list of strings")
    else:
        leaked = sorted(t for t in td if is_public_benchmark(t))
        if leaked:
            errors.append(f"train_domains must not include public-benchmark eval sources: {leaked}")

    if not _nonempty_str_list(d.get("eval_sets")):
        errors.append("'eval_sets' must be a non-empty list of strings")

    sc = d.get("success_criteria")
    if not isinstance(sc, dict) or not sc:
        errors.append("'success_criteria' must be a non-empty object")
    else:
        for k, v in sc.items():
            if not _is_number(v):
                errors.append(f"success_criteria.{k} must be numeric")

    ncp = d.get("near_ceiling_eval_policy")
    if not isinstance(ncp, dict) or not ncp:
        errors.append("'near_ceiling_eval_policy' must be a non-empty object")
    else:
        tol = ncp.get("use_do_not_regress_tolerance")
        if not _is_number(tol):
            errors.append("near_ceiling_eval_policy.use_do_not_regress_tolerance must be numeric")
        elif not (-0.02 <= tol <= 0):
            errors.append("near_ceiling_eval_policy.use_do_not_regress_tolerance must be in [-0.02, 0]")
        oracle = ncp.get("if_oracle_ndcg10_ge")
        if not _is_number(oracle) or not (0 < oracle <= 1):
            errors.append("near_ceiling_eval_policy.if_oracle_ndcg10_ge must be a number in (0, 1]")
        if ncp.get("do_not_use_as_primary_promotion_signal") is not True:
            errors.append("near_ceiling_eval_policy.do_not_use_as_primary_promotion_signal MUST be true")

    return errors


def load_v5_rag_config(path: str | Path) -> V5RagConfig:
    d = _read(path)
    errors = validate_v5_rag(d)
    if errors:
        raise ValueError("invalid v5 RAG config: " + "; ".join(errors))
    return V5RagConfig(
        experiment_id=d["experiment_id"], goal=d["goal"],
        legal_eval_is_diagnostic_only=bool(d["legal_eval_is_diagnostic_only"]),
        public_benchmarks_eval_only=bool(d["public_benchmarks_eval_only"]),
        dense_candidates=list(d["dense_candidates"]),
        reranker_candidates=list(d["reranker_candidates"]),
        teacher_models=dict(d["teacher_models"]),
        train_domains=list(d["train_domains"]), eval_sets=list(d["eval_sets"]),
        success_criteria=dict(d["success_criteria"]),
        near_ceiling_eval_policy=dict(d["near_ceiling_eval_policy"]), raw=d)
