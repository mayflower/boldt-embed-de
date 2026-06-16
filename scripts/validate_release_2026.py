#!/usr/bin/env python3
"""2026 release gate: refuse to ship over-claimed or unprovenanced artifacts (Prompt 12).

Blocking checks (pure-stdlib, so each is unit-testable on fixtures):

* required 2026 configs exist (teacher/student/baseline + training/eval),
* no model-weight files or checkpoints are committed,
* no teacher cache is committed under outputs/teacher-cache/,
* model cards carry the required provenance/limitation sections + a non-legal-advice
  warning, and contain no banned overclaim phrases,
* RELEASE_CHECKLIST references the experiment run cards.

This is separate from `validate_repo.py` (structure/JSON/imports). `make validate` stays
green; this gate is run before a release.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]

WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".gguf")
BANNED_PHRASES = ["state-of-the-art", "best-in-class", "outperforms all", "world-class",
                  "unbeatable", "guaranteed results", "no. 1 ", "#1 embedding"]
REQUIRED_CONFIGS = [
    "configs/teacher_models.json", "configs/student_training_2026.json",
    "configs/baseline_models.json", "configs/training_causal.json",
    "configs/training_bidirectional.json", "configs/training_reranker.json",
    "configs/evaluation.json", "configs/data_sources_v2.json",
    "configs/experiments/v2_generalization.json",
]
CARD_COMMON_SECTIONS = ["## Teacher distillation", "## Training data provenance",
                        "## Leakage policy", "## German stress tests", "## Limitations",
                        "## Production default", "## Known failure modes"]
# v2 artifacts required only for a RELEASE-readiness check (--require-v2-artifacts).
V2_REQUIRED_ARTIFACTS = [
    "candidates_v2.report.json", "teacher-cache/qwen3_v2.summary.json",
    "eval/dense_germanquad.json", "eval/dense_dt_test.json", "eval/dense_gerdalir.json",
    "real_matryoshka_germanquad.json", "reranker-lift-germanquad-v2.json",
]
EMBEDDER_EXTRA = ["## Matryoshka dimensions"]
RERANKER_EXTRA = ["## Reranker lift"]
NON_LEGAL_WARNING = "not legal advice"
# v4 RAG reranker track (no legal/admin requirement). Paths are relative to --results-dir.
V4_RAG_REQUIRED_ARTIFACTS = [
    "eval/webfaq/queries.jsonl",                       # WebFAQ held-out eval split
    "candidate_lists/rag_reranker_train_lists.jsonl",  # fixed candidate lists
    "teacher/rag_train_scored.jsonl",                  # teacher-scored lists
    "eval/reranker_lift_webfaq.json",                  # reranker lift reports
    "eval/rag_reranker_gate.json",                     # promotion gate
]
V4_RAG_RECOMMENDED_PHRASE = "Recommended for German FAQ/RAG reranking"
V4_RAG_CARD_DISCLAIMERS = ["not legal advice", "not a dense retriever",
                           "candidate lists only", "lift over"]

# SCOPE RESET (v6): the product is a Boldt dense German RAG embedder + a STANDALONE reranker, each
# measured DIRECTLY under the harness. Policy-gated serving (rerank-or-abstain / bounded
# margin_override) is DIAGNOSTIC ONLY and must NEVER be recommended as a production workaround.
# Reranker promotion requires RAW reranker lift over FIXED candidate lists (the raw v4/v5 gates);
# policy-gated variants do not count for model promotion.
V5_SMALL_RAG_REQUIRED_ARTIFACTS = [
    "V5_RESULTS.json",                 # honest run summary
    "eval/v5_rag_lift_gate.json",      # raw (always-rerank) hardness gate
    "abstain/fit_report.json",         # policy fit (dev only) — diagnostic
    "abstain/eval_webfaq.json",        # policy eval — diagnostic
    "abstain/eval_germanquad.json",    # policy eval — diagnostic
    "abstain/eval_dt_test.json",       # policy eval — diagnostic
    "abstain/gate.json",               # policy promotion gate — diagnostic
]
# A RAW reranker recommendation (lift over FIXED candidate lists) is allowed ONLY if the RAW gate
# passes — never via a serving policy.
V5_RAW_RECOMMENDED_PHRASE = "Recommended for German RAG reranking over fixed candidate lists"
RAW_RECOMMENDED_PHRASES = ("Recommended for German FAQ/RAG reranking",   # v4 raw-lift phrase
                           V5_RAW_RECOMMENDED_PHRASE)                    # v5 raw-lift phrase
# Kept as a KNOWN-BANNED example: a policy-gated serving recommendation, never allowed.
V5_RECOMMENDED_PHRASE = "Recommended for German RAG reranking with the abstention policy"
POLICY_SERVING_TOKENS = ("policy-gated", "policy gated", "bounded policy", "bounded margin_override",
                         "bounded margin override", "bounded rerank", "abstention policy",
                         "abstain policy", "rerank-or-abstain", "rerank or abstain", "serving policy",
                         "margin_override policy")
RECOMMEND_TOKENS = ("recommend", "production default", "production-ready", "production ready",
                    "promote to production")
# A policy line is fine when it is negated or explicitly framed as diagnostics/analysis.
DIAGNOSTIC_NEGATION = ("not ", "never", "no ", "without", "diagnostic", "analysis only",
                       "only diagnostic", "experimental", "do not", "don't", "cannot", "isn't")
BANNED_POLICY_RECOMMENDATION_PHRASES = (
    V5_RECOMMENDED_PHRASE.lower(), "recommended with the abstention policy",
    "recommended with policy", "recommended with the bounded policy",
    "recommended via the bounded policy", "policy-gated reranker is recommended",
    "recommended as a policy-gated",
)
# v6 active RAG track: a Boldt DENSE embedder + a RAW reranker, each gated on its OWN gate. The
# dense embedder may be recommended only if the dense-recall gate passes AND recall/eval reports
# exist; the reranker only if the RAW reranker gate passes. GerDaLIR/legal is diagnostic-only.
V6_DENSE_RECOMMENDED_PHRASE = "Recommended for German RAG first-stage retrieval"
V6_RAW_RECOMMENDED_PHRASE = "Recommended for German RAG reranking (raw, over fixed candidate lists)"
V6_DENSE_DIR = ("outputs", "v6-dense-rag")
V6_RERANKER_DIR = ("outputs", "v6-reranker")
V6_DENSE_REQUIRED_ARTIFACTS = ["dense_recall_gate.json", "webfaq_real_recall_bm25_vs_dense.json",
                               "first_stage_audit_webfaq.json"]
V6_RAW_RERANKER_REQUIRED_ARTIFACTS = ["raw_gate.json", "eval/webfaq_lift.json",
                                      "eval/germanquad_lift.json", "eval/dt_test_lift.json"]
V6_RAG_EVAL_SETS = ("webfaq", "local_rag", "germanquad", "dt_test")   # gerdalir/legal = diagnostic
POLICY_RESULT_MODE_TOKENS = ("bounded", "policy", "abstain", "margin_override")

CARD_TYPES = {
    "Boldt-Embed-DE-350M-v1-causal.md": "embedder",
    "Boldt-Embed-DE-350M-v1-bi.md": "embedder",
    "Boldt-Reranker-DE-350M-v1.md": "reranker",
}

Issue = Tuple[str, str]


# --------------------------------------------------------------------- pure checks
def check_no_committed_weights(tracked_files: Sequence[str]) -> List[Issue]:
    out = []
    for f in tracked_files:
        low = f.lower()
        if low.endswith(WEIGHT_EXTS):
            out.append(("committed_weight_file", f))
        elif "checkpoints/" in f and not f.startswith("tests/"):
            out.append(("committed_checkpoint", f))
    return out


def check_no_committed_teacher_cache(tracked_files: Sequence[str]) -> List[Issue]:
    return [("committed_teacher_cache", f) for f in tracked_files
            if f.startswith("outputs/teacher-cache/")]


def check_required_configs(root: Path) -> List[Issue]:
    return [("missing_config", c) for c in REQUIRED_CONFIGS if not (root / c).exists()]


def check_overclaims(name: str, text: str) -> List[Issue]:
    low = text.lower()
    return [("overclaim", f"{name}: '{p}'") for p in BANNED_PHRASES if p in low]


def check_card_sections(name: str, text: str, card_type: str) -> List[Issue]:
    required = list(CARD_COMMON_SECTIONS)
    required += EMBEDDER_EXTRA if card_type == "embedder" else RERANKER_EXTRA
    issues = [("card_missing_section", f"{name}: {s}") for s in required if s not in text]
    if NON_LEGAL_WARNING not in text.lower():
        issues.append(("card_missing_non_legal_warning", name))
    return issues


def check_checklist_references_runcards(text: str) -> List[Issue]:
    low = text.lower()
    if "run card" not in low and "experiments.md" not in low:
        return [("checklist_no_runcard_reference", "RELEASE_CHECKLIST.md")]
    return []


def check_v2_manifest(root: Path) -> List[Issue]:
    """v2 source manifest exists and validates (public benchmarks blocked from training)."""
    sys.path.insert(0, str(root / "src"))
    from boldt_embed import source_manifest as sm
    path = root / "configs" / "data_sources_v2.json"
    if not path.exists():
        return [("missing", "configs/data_sources_v2.json")]
    d = json.loads(path.read_text(encoding="utf-8"))
    return [("v2_manifest", e) for e in sm.validate_source_manifest(d)]


def check_benchmark_eval_only(root: Path) -> List[Issue]:
    """Benchmark tasks are eval-only and no eval dataset is training-allowed in the manifest."""
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))
    from boldt_embed import source_manifest as sm
    import run_baseline_benchmarks as rb
    tasks_path = root / "benchmarks" / "mteb_german_tasks.json"
    man_path = root / "configs" / "data_sources_v2.json"
    if not tasks_path.exists():
        return [("missing", "benchmarks/mteb_german_tasks.json")]
    tg = rb.load_benchmark_tasks(tasks_path)
    issues = [("benchmark_task", e) for e in rb.validate_benchmark_tasks(tg)]
    if man_path.exists():
        entries = sm.load_source_manifest(man_path)
        issues += [("eval_leakage", e) for e in rb.check_eval_leakage_against_manifest(tg, entries)]
    return issues


def _lift_delta(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    fs = next((v for k, v in d.items() if k.startswith("first_stage_ndcg@")), None)
    rr = next((v for k, v in d.items() if k.startswith("student_reranker_ndcg@")), None)
    return None if fs is None or rr is None else round(rr - fs, 4)


def check_reranker_promotion(results_dir: Path) -> List[Issue]:
    """If v2 reranker lift reports exist, the reranker must NOT degrade any held-out set."""
    issues = []
    for ds in ("germanquad", "dt_test"):
        p = results_dir / f"reranker-lift-{ds}-v2.json"
        if p.exists():
            delta = _lift_delta(p)
            if delta is not None and delta < 0.0:
                issues.append(("reranker_degrades", f"{ds}: delta {delta} < 0 — promotion gate fails"))
    return issues


def check_v2_artifacts(results_dir: Path) -> List[Issue]:
    """RELEASE-readiness: every required v2 run artifact must exist."""
    return [("missing_v2_artifact", a) for a in V2_REQUIRED_ARTIFACTS
            if not (results_dir / a).exists()]


def check_v4_rag_artifacts(root: Path, results_dir: Path) -> List[Issue]:
    """v4 RAG-reranker readiness: config + WebFAQ eval split + fixed candidate lists +
    teacher-scored lists + lift reports + promotion gate. NO legal/admin requirement."""
    issues: List[Issue] = []
    if not (root / "configs" / "experiments" / "v4_rag_reranker.json").exists():
        issues.append(("missing_config", "configs/experiments/v4_rag_reranker.json"))
    issues += [("missing_v4_rag_artifact", a) for a in V4_RAG_REQUIRED_ARTIFACTS
               if not (results_dir / a).exists()]
    return issues


def _v4_gate_passed(results_dir: Path) -> Optional[bool]:
    p = results_dir / "eval" / "rag_reranker_gate.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("status") == "pass"
    except Exception:
        return None


def check_v4_rag_card(root: Path, results_dir: Path) -> List[Issue]:
    """The reranker card may claim the RAG recommendation ONLY if the v4 promotion gate passed,
    and it must always carry the v4 disclaimers. Legal/admin are NOT required for this track."""
    issues: List[Issue] = []
    card = root / "model_cards" / "Boldt-Reranker-DE-350M-v1.md"
    if not card.exists():
        return [("missing_card", "Boldt-Reranker-DE-350M-v1.md")]
    text = card.read_text(encoding="utf-8")
    low = text.lower()
    for phrase in V4_RAG_CARD_DISCLAIMERS:
        if phrase.lower() not in low:
            issues.append(("v4_card_missing_disclaimer", phrase))
    if V4_RAG_RECOMMENDED_PHRASE in text and _v4_gate_passed(results_dir) is not True:
        issues.append(("v4_card_recommended_without_passing_gate",
                       "card claims the RAG recommendation but the v4 promotion gate did not pass"))
    return issues


def check_v5_small_rag_artifacts(root: Path) -> List[Issue]:
    """v5 small-RAG readiness: config + run summary + raw gate + abstain fit/eval/gate artifacts.
    No legal/admin requirement; guardrails are eval-only."""
    issues: List[Issue] = []
    if not (root / "configs" / "experiments" / "v5_small_rag.json").exists():
        issues.append(("missing_config", "configs/experiments/v5_small_rag.json"))
    v5 = root / "outputs" / "v5-small-rag"
    issues += [("missing_v5_small_rag_artifact", a) for a in V5_SMALL_RAG_REQUIRED_ARTIFACTS
               if not (v5 / a).exists()]
    return issues


def check_no_policy_gated_recommendation(name: str, text: str) -> List[Issue]:
    """A model card must NEVER recommend a policy-gated serving workaround (rerank-or-abstain /
    bounded margin_override). Policy work may be MENTIONED only as diagnostics/analysis — a line that
    is negated or explicitly diagnostic is fine. Pure function (testable on text)."""
    issues: List[Issue] = []
    seen: set = set()
    low = text.lower()
    for p in BANNED_POLICY_RECOMMENDATION_PHRASES:
        if p in low and p not in seen:
            seen.add(p)
            issues.append(("card_recommends_policy_gated_serving", f"{name}: '{p}'"))
    for raw_line in text.split("\n"):
        line = raw_line.lower()
        if (any(t in line for t in POLICY_SERVING_TOKENS) and any(r in line for r in RECOMMEND_TOKENS)
                and not any(nz in line for nz in DIAGNOSTIC_NEGATION)):
            key = raw_line.strip()[:100]
            if key not in seen:
                seen.add(key)
                issues.append(("card_recommends_policy_gated_serving", f"{name}: {key}"))
    return issues


def check_reranker_raw_recommendation(text: str, *, v4_gate_passed: Optional[bool],
                                      v5_raw_gate_passed: Optional[bool]) -> List[Issue]:
    """A RAW reranker recommendation (lift over FIXED candidate lists) is allowed ONLY if the
    corresponding RAW gate passed. While a raw gate fails, the reranker stays NOT recommended."""
    issues: List[Issue] = []
    if "Recommended for German FAQ/RAG reranking" in text and v4_gate_passed is not True:
        issues.append(("raw_reranker_recommended_without_passing_gate",
                       "v4 FAQ/RAG raw recommendation but the v4 raw gate did not pass"))
    if V5_RAW_RECOMMENDED_PHRASE in text and v5_raw_gate_passed is not True:
        issues.append(("raw_reranker_recommended_without_passing_gate",
                       "v5 raw recommendation but the v5 raw gate did not pass"))
    return issues


def _v5_raw_gate_passed(root: Path) -> Optional[bool]:
    p = root / "outputs" / "v5-small-rag" / "eval" / "v5_rag_lift_gate.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("status") == "pass"
    except Exception:
        return None


def check_v5_small_rag_card(root: Path) -> List[Issue]:
    """v5 reranker card: policy-gated serving (incl. the abstention recommendation) is NEVER
    allowed (diagnostics only); always-rerank must never be recommended; and a RAW v5 recommendation
    is allowed only if the RAW v5 gate passed."""
    issues: List[Issue] = []
    card = root / "model_cards" / "Boldt-Reranker-DE-350M-v1.md"
    if not card.exists():
        return [("missing_card", "Boldt-Reranker-DE-350M-v1.md")]
    text = card.read_text(encoding="utf-8")
    for _k, d in check_no_policy_gated_recommendation("Boldt-Reranker-DE-350M-v1.md", text):
        issues.append(("v5_card_recommends_policy_gated_serving", d))
    for line in text.lower().split("\n"):
        if "always-rerank" in line and "recommend" in line and "not" not in line and "never" not in line:
            issues.append(("v5_card_recommends_always_rerank", line.strip()[:80]))
            break
    if V5_RAW_RECOMMENDED_PHRASE in text and _v5_raw_gate_passed(root) is not True:
        issues.append(("v5_card_raw_recommended_without_passing_gate",
                       "card claims a raw v5 RAG recommendation but the raw v5 gate did not pass"))
    return issues


# --------------------------------------------------------------- v6 active product track
def _gate_status_pass(path: Path) -> Optional[bool]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "pass"
    except Exception:
        return None


def _v6_dense_gate_passed(root: Path) -> Optional[bool]:
    return _gate_status_pass(root.joinpath(*V6_DENSE_DIR) / "dense_recall_gate.json")


def _v6_raw_reranker_gate_passed(root: Path) -> Optional[bool]:
    return _gate_status_pass(root.joinpath(*V6_RERANKER_DIR) / "raw_gate.json")


def check_v6_dense_recommendation(root: Path) -> List[Issue]:
    """A dense-embedder card may claim the v6 retrieval recommendation ONLY if the dense-recall gate
    passed AND the recall/eval reports exist (provenance). Rule 1 + 'no dense rec without reports'."""
    issues: List[Issue] = []
    dense_dir = root.joinpath(*V6_DENSE_DIR)
    recall_report = dense_dir / "webfaq_real_recall_bm25_vs_dense.json"
    for fname in ("Boldt-Embed-DE-350M-v1-causal.md", "Boldt-Embed-DE-350M-v1-bi.md"):
        card = root / "model_cards" / fname
        if not card.exists():
            continue
        if V6_DENSE_RECOMMENDED_PHRASE in card.read_text(encoding="utf-8"):
            if not recall_report.exists():
                issues.append(("dense_recommended_without_recall_reports", fname))
            if _v6_dense_gate_passed(root) is not True:
                issues.append(("dense_recommended_without_passing_dense_gate", fname))
    return issues


def check_v6_raw_reranker_recommendation(root: Path) -> List[Issue]:
    """The reranker card may claim the v6 RAW recommendation ONLY if the v6 RAW reranker gate passed."""
    card = root / "model_cards" / "Boldt-Reranker-DE-350M-v1.md"
    if not card.exists():
        return []
    if V6_RAW_RECOMMENDED_PHRASE in card.read_text(encoding="utf-8") \
            and _v6_raw_reranker_gate_passed(root) is not True:
        return [("v6_reranker_recommended_without_passing_raw_gate",
                 "card claims the v6 raw recommendation but the v6 raw reranker gate did not pass")]
    return []


def check_no_policy_result_as_promotion_evidence(root: Path) -> List[Issue]:
    """No policy-gated/bounded/abstain result may serve as promotion evidence: every v6 eval-lift
    report and the raw gate must be evaluated in 'raw' ranking mode."""
    issues: List[Issue] = []
    eval_dir = root.joinpath(*V6_RERANKER_DIR) / "eval"
    if eval_dir.exists():
        for p in sorted(eval_dir.glob("*_lift.json")):
            try:
                mode = str(json.loads(p.read_text(encoding="utf-8")).get("ranking_mode", "")).lower()
            except Exception:
                continue
            if mode and (mode != "raw" or any(t in mode for t in POLICY_RESULT_MODE_TOKENS)):
                issues.append(("policy_result_as_promotion_evidence", f"{p.name}: ranking_mode={mode}"))
    gate = root.joinpath(*V6_RERANKER_DIR) / "raw_gate.json"
    if gate.exists():
        try:
            g = json.loads(gate.read_text(encoding="utf-8"))
            if str(g.get("evaluated_ranking_mode", "raw")).lower() != "raw" \
                    or g.get("policy_gated_result_used") is True:
                issues.append(("policy_result_as_promotion_evidence", "raw_gate.json not raw-only"))
        except Exception:
            pass
    return issues


def check_no_public_eval_leakage_v6(root: Path) -> List[Issue]:
    """Surface public-eval leakage flagged by the v6 gates (in addition to the manifest-level check)."""
    issues: List[Issue] = []
    for gate_path in (root.joinpath(*V6_RERANKER_DIR) / "raw_gate.json",):
        if not gate_path.exists():
            continue
        try:
            g = json.loads(gate_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for c in g.get("checks", []):
            if c.get("check") == "no_public_eval_leakage" and c.get("status") == "fail":
                issues.append(("public_eval_leakage", f"{gate_path.name}: {c.get('detail')}"))
    return issues


def check_v6_dense_artifacts(root: Path) -> List[Issue]:
    d = root.joinpath(*V6_DENSE_DIR)
    return [("missing_v6_dense_artifact", a) for a in V6_DENSE_REQUIRED_ARTIFACTS
            if not (d / a).exists()]


def check_v6_raw_reranker_artifacts(root: Path) -> List[Issue]:
    d = root.joinpath(*V6_RERANKER_DIR)
    return [("missing_v6_raw_reranker_artifact", a) for a in V6_RAW_RERANKER_REQUIRED_ARTIFACTS
            if not (d / a).exists()]


def _summary_unknown_license_rows(summary: Dict[str, object]) -> int:
    """Unknown-license count, tolerant of old (by_license only) and new (explicit) schemas."""
    if "unknown_license_rows" in summary:
        try:
            return int(summary["unknown_license_rows"])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0
    by_lic = summary.get("by_license") or {}
    if isinstance(by_lic, dict):
        return int(by_lic.get("unknown", 0) or 0)
    return 0


def check_teacher_cache_license(results_dir: Path) -> List[Issue]:
    """Every teacher-cache summary under results_dir must have ZERO unknown-license rows.
    This blocks the v2 provenance bug (`by_license {"unknown": N}`) from reaching release."""
    issues: List[Issue] = []
    summaries = sorted(results_dir.glob("**/*.summary.json"))
    for sp in summaries:
        try:
            summary = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(summary, dict) or "by_license" not in summary:
            continue  # not a teacher-cache summary
        n = _summary_unknown_license_rows(summary)
        if n > 0:
            issues.append(("teacher_cache_unknown_license",
                           f"{sp.relative_to(results_dir)}: {n} unknown-license rows"))
        if int((summary.get("disallowed_for_training_rows") or 0)) > 0:
            issues.append(("teacher_cache_disallowed_training",
                           f"{sp.relative_to(results_dir)}: "
                           f"{summary['disallowed_for_training_rows']} disallowed-training rows"))
    return issues


# ------------------------------------------------------------------------- runner
def git_tracked_files(root: Path) -> List[str]:
    try:
        out = subprocess.check_output(["git", "ls-files"], cwd=root, text=True)
        return [line for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def run_checks(root: Path = ROOT, results_dir: Path = None,
               require_v2_artifacts: bool = False,
               require_v3_artifacts: bool = False,
               require_v4_rag_artifacts: bool = False,
               require_v5_small_rag_artifacts: bool = False,
               require_v6_dense_artifacts: bool = False,
               require_v6_raw_reranker_artifacts: bool = False) -> Dict[str, object]:
    tracked = git_tracked_files(root)
    results_dir = results_dir or (root / "outputs")
    checks: Dict[str, List[Issue]] = {
        "required_configs": check_required_configs(root),
        "no_committed_weights": check_no_committed_weights(tracked),
        "no_committed_teacher_cache": check_no_committed_teacher_cache(tracked),
        "v2_manifest": check_v2_manifest(root),
        "benchmark_eval_only": check_benchmark_eval_only(root),
        "reranker_promotion": check_reranker_promotion(results_dir),
        "model_cards": [],
    }
    checks["policy_gated_serving"] = []          # always-on: never recommend a serving policy
    for fname, ctype in CARD_TYPES.items():
        path = root / "model_cards" / fname
        if not path.exists():
            checks["model_cards"].append(("missing_card", fname))
            continue
        text = path.read_text(encoding="utf-8")
        checks["model_cards"] += check_overclaims(fname, text)
        checks["model_cards"] += check_card_sections(fname, text, ctype)
        checks["policy_gated_serving"] += check_no_policy_gated_recommendation(fname, text)
    # always-on: a RAW reranker recommendation is gated on the RAW gates (policy variants don't count)
    rcard = root / "model_cards" / "Boldt-Reranker-DE-350M-v1.md"
    if rcard.exists():
        checks["reranker_raw_recommendation"] = check_reranker_raw_recommendation(
            rcard.read_text(encoding="utf-8"),
            v4_gate_passed=_v4_gate_passed(root / "outputs" / "v4-rag-reranker"),
            v5_raw_gate_passed=_v5_raw_gate_passed(root))
    # always-on v6 active-track enforcement: recommendations gated on the real gates; no policy
    # result as promotion evidence; surface public-eval leakage flagged by the v6 gates.
    checks["v6_dense_recommendation"] = check_v6_dense_recommendation(root)
    checks["v6_raw_reranker_recommendation"] = check_v6_raw_reranker_recommendation(root)
    checks["no_policy_result_as_promotion_evidence"] = check_no_policy_result_as_promotion_evidence(root)
    checks["public_eval_leakage_v6"] = check_no_public_eval_leakage_v6(root)
    checklist = root / "RELEASE_CHECKLIST.md"
    checks["checklist"] = (check_checklist_references_runcards(checklist.read_text(encoding="utf-8"))
                           if checklist.exists() else [("missing", "RELEASE_CHECKLIST.md")])
    if require_v2_artifacts:
        checks["v2_artifacts"] = check_v2_artifacts(results_dir)
    # A v2- OR v3-readiness run must be license-clean: no unknown-license teacher-cache rows.
    if require_v2_artifacts or require_v3_artifacts:
        checks["teacher_cache_license"] = check_teacher_cache_license(results_dir)
    # v4 RAG reranker track: artifacts + card-vs-gate consistency. NO legal/admin requirement.
    if require_v4_rag_artifacts:
        v4_dir = results_dir if (results_dir / "eval" / "rag_reranker_gate.json").exists() \
            else (root / "outputs" / "v4-rag-reranker")
        checks["v4_rag_artifacts"] = check_v4_rag_artifacts(root, v4_dir)
        checks["v4_rag_card"] = check_v4_rag_card(root, v4_dir)
    # v5 small-RAG track: artifacts + abstention-policy card-vs-gate consistency.
    if require_v5_small_rag_artifacts:
        checks["v5_small_rag_artifacts"] = check_v5_small_rag_artifacts(root)
        checks["v5_small_rag_card"] = check_v5_small_rag_card(root)
    # v6 active track: dense embedder + RAW reranker artifact-readiness (RELEASE gate).
    if require_v6_dense_artifacts:
        checks["v6_dense_artifacts"] = check_v6_dense_artifacts(root)
    if require_v6_raw_reranker_artifacts:
        checks["v6_raw_reranker_artifacts"] = check_v6_raw_reranker_artifacts(root)
    issues = [i for group in checks.values() for i in group]
    return {"status": "pass" if not issues else "fail", "issue_count": len(issues),
            "checks": {k: [list(i) for i in v] for k, v in checks.items()}}


def render_markdown(report: Dict[str, object]) -> str:
    lines = ["# Boldt-Embed-DE 2026 Release Gate", "",
             f"Status: **{report['status']}**", f"Issue count: {report['issue_count']}", ""]
    for name, issues in report["checks"].items():  # type: ignore[union-attr]
        lines.append(f"## {name}")
        lines.append("PASS" if not issues else "\n".join(f"- {k}: {d}" for k, d in issues))
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", choices=["json", "markdown"], default="json")
    ap.add_argument("--results-dir", default=None, help="dir of v2 run artifacts (reranker lift, etc.)")
    ap.add_argument("--require-v2-artifacts", action="store_true",
                    help="RELEASE gate: also require all v2 run artifacts to exist")
    ap.add_argument("--require-v3-artifacts", action="store_true",
                    help="RELEASE gate: enforce license-clean teacher cache (zero unknown-license rows)")
    ap.add_argument("--require-v4-rag-artifacts", action="store_true",
                    help="RELEASE gate: v4 RAG reranker artifacts + card-vs-gate (no legal/admin requirement)")
    ap.add_argument("--require-v5-small-rag-artifacts", action="store_true",
                    help="RELEASE gate: v5 small-RAG artifacts + abstention-policy card-vs-gate")
    ap.add_argument("--require-v6-dense-artifacts", action="store_true",
                    help="RELEASE gate: v6 dense-embedder recall/eval reports + dense-recall gate")
    ap.add_argument("--require-v6-raw-reranker-artifacts", action="store_true",
                    help="RELEASE gate: v6 RAW reranker gate + raw lift reports (no policy)")
    args = ap.parse_args()
    report = run_checks(results_dir=Path(args.results_dir) if args.results_dir else None,
                        require_v2_artifacts=args.require_v2_artifacts,
                        require_v3_artifacts=args.require_v3_artifacts,
                        require_v4_rag_artifacts=args.require_v4_rag_artifacts,
                        require_v5_small_rag_artifacts=args.require_v5_small_rag_artifacts,
                        require_v6_dense_artifacts=args.require_v6_dense_artifacts,
                        require_v6_raw_reranker_artifacts=args.require_v6_raw_reranker_artifacts)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.format == "json"
          else render_markdown(report))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
