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
from typing import Dict, List, Sequence, Tuple

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


# ------------------------------------------------------------------------- runner
def git_tracked_files(root: Path) -> List[str]:
    try:
        out = subprocess.check_output(["git", "ls-files"], cwd=root, text=True)
        return [line for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def run_checks(root: Path = ROOT, results_dir: Path = None,
               require_v2_artifacts: bool = False) -> Dict[str, object]:
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
    args = ap.parse_args()
    report = run_checks(results_dir=Path(args.results_dir) if args.results_dir else None,
                        require_v2_artifacts=args.require_v2_artifacts)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.format == "json"
          else render_markdown(report))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
