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

# v5 small-RAG: the reranker may be recommended ONLY with the abstention policy AND only if the
# v5 abstain gate passes. always-rerank must NEVER be recommended.
V5_SMALL_RAG_REQUIRED_ARTIFACTS = [
    "V5_RESULTS.json",                 # honest run summary
    "eval/v5_rag_lift_gate.json",      # raw (always-rerank) hardness gate
    "abstain/fit_report.json",         # policy fit (dev only)
    "abstain/eval_webfaq.json",        # policy eval — primary
    "abstain/eval_germanquad.json",    # policy eval — guardrail
    "abstain/eval_dt_test.json",       # policy eval — guardrail
    "abstain/gate.json",               # policy promotion gate
]
V5_RECOMMENDED_PHRASE = "Recommended for German RAG reranking with the abstention policy"
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


def _v5_abstain_gate_passed(root: Path) -> Optional[bool]:
    p = root / "outputs" / "v5-small-rag" / "abstain" / "gate.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("status") == "pass"
    except Exception:
        return None


def check_v5_small_rag_card(root: Path) -> List[Issue]:
    """The reranker card may claim the v5 abstention recommendation ONLY if the v5 abstain gate
    passed, and must NEVER recommend always-rerank."""
    issues: List[Issue] = []
    card = root / "model_cards" / "Boldt-Reranker-DE-350M-v1.md"
    if not card.exists():
        return [("missing_card", "Boldt-Reranker-DE-350M-v1.md")]
    text = card.read_text(encoding="utf-8")
    if V5_RECOMMENDED_PHRASE in text and _v5_abstain_gate_passed(root) is not True:
        issues.append(("v5_card_recommended_without_passing_gate",
                       "card claims the v5 abstention recommendation but the v5 gate did not pass"))
    low = text.lower()
    if "always-rerank" in low and "recommend" in low:
        # crude co-occurrence guard: never recommend always-rerank
        for line in low.splitlines():
            if "always-rerank" in line and "recommend" in line and "not" not in line and "never" not in line:
                issues.append(("v5_card_recommends_always_rerank", line.strip()[:80]))
                break
    return issues


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
               require_v5_small_rag_artifacts: bool = False) -> Dict[str, object]:
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
    for fname, ctype in CARD_TYPES.items():
        path = root / "model_cards" / fname
        if not path.exists():
            checks["model_cards"].append(("missing_card", fname))
            continue
        text = path.read_text(encoding="utf-8")
        checks["model_cards"] += check_overclaims(fname, text)
        checks["model_cards"] += check_card_sections(fname, text, ctype)
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
    args = ap.parse_args()
    report = run_checks(results_dir=Path(args.results_dir) if args.results_dir else None,
                        require_v2_artifacts=args.require_v2_artifacts,
                        require_v3_artifacts=args.require_v3_artifacts,
                        require_v4_rag_artifacts=args.require_v4_rag_artifacts,
                        require_v5_small_rag_artifacts=args.require_v5_small_rag_artifacts)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.format == "json"
          else render_markdown(report))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
