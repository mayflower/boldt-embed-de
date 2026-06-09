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
    "configs/evaluation.json",
]
CARD_COMMON_SECTIONS = ["## Teacher distillation", "## Training data provenance",
                        "## Leakage policy", "## German stress tests", "## Limitations"]
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


# ------------------------------------------------------------------------- runner
def git_tracked_files(root: Path) -> List[str]:
    try:
        out = subprocess.check_output(["git", "ls-files"], cwd=root, text=True)
        return [line for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def run_checks(root: Path = ROOT) -> Dict[str, object]:
    tracked = git_tracked_files(root)
    checks: Dict[str, List[Issue]] = {
        "required_configs": check_required_configs(root),
        "no_committed_weights": check_no_committed_weights(tracked),
        "no_committed_teacher_cache": check_no_committed_teacher_cache(tracked),
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
    args = ap.parse_args()
    report = run_checks()
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.format == "json"
          else render_markdown(report))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
