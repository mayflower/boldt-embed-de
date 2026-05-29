#!/usr/bin/env python3
"""Structural validation for the Boldt-Embed-DE implementation repo.

Runs on the Python standard library only. It checks that the repository structure is
present and well-formed, that all JSON parses, that the package imports, and that any
ADRs / model cards that exist contain their required sections. It validates what is
present; it never fabricates results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

REQUIRED_DIRS = [
    "src/boldt_embed",
    "configs",
    "scripts",
    "tests",
    "docs",
    "docs/adr",
    "benchmarks",
    "data",
    "data/schema",
    "data/samples",
    "data/synthetic",
    "schemas",
    "docs/data",
    "model_cards",
    "outputs",
]

_SRC_MODULES = [
    "__init__", "textutil", "config", "instructions", "pooling", "matryoshka",
    "losses", "merging", "metrics", "data", "hard_negatives", "eval_harness",
    "model_causal", "model_bidirectional", "reranker", "cli", "train",
]
_SCRIPTS = [
    "validate_repo", "run_smoke_tests", "run_local_benchmark",
    "run_mteb_benchmark_template", "write_reports", "run_real_training",
    "run_real_bidirectional", "run_real_reranker",
    "validate_data_schema", "export_sentence_transformers",
    "train_causal", "train_bidirectional", "train_reranker",
]
_DOCS = [
    "RESEARCH_NOTES_2026", "ARCHITECTURE_PLAN", "DATA_PLAN",
    "SYNTHETIC_PAIRS", "VALIDATION_POLICY", "BENCHMARK_PLAN",
]
_ADRS = [
    "ADR-001-base-model-and-license", "ADR-002-causal-vs-bidirectional",
    "ADR-003-pooling-strategy", "ADR-004-training-data-and-licensing",
    "ADR-005-benchmark-protocol", "ADR-006-release-and-model-card",
    "ADR-007-matryoshka-dimensions", "ADR-008-reranker-architecture",
    "ADR-009-training-evaluation-split",
]

REQUIRED_FILES = [
    "README.md", "CLAUDE.md", "LICENSE", "pyproject.toml", "Makefile",
    "configs/training_causal.json", "configs/training_bidirectional.json",
    "configs/training_reranker.json", "configs/evaluation.json",
    "benchmarks/toy_de_retrieval.json", "benchmarks/stress_cases_de.jsonl",
    "benchmarks/mteb_german_tasks.json", "benchmarks/baselines.json",
    "data/schema/pair_schema.json", "data/samples/toy_pairs_de.jsonl",
    "data/samples/toy_triples_de.jsonl", "data/synthetic/prompt_specs.json",
]
REQUIRED_FILES += [f"src/boldt_embed/{m}.py" for m in _SRC_MODULES]
REQUIRED_FILES += [f"scripts/{s}.py" for s in _SCRIPTS]
REQUIRED_FILES += [f"docs/{d}.md" for d in _DOCS]
REQUIRED_FILES += ["docs/research/llm2embed-2026.md"]
REQUIRED_FILES += [
    "schemas/training_pair.schema.json", "schemas/benchmark_result.schema.json",
    "docs/data/data-sources.md", "docs/data/license-policy.md", "docs/data/leakage-policy.md",
]
REQUIRED_FILES += [f"docs/adr/{a}.md" for a in _ADRS]
REQUIRED_FILES += [
    "RELEASE_CHECKLIST.md",
    "AUDIT.md",
    "model_cards/Boldt-Embed-DE-350M-v1-causal.md",
    "model_cards/Boldt-Embed-DE-350M-v1-bi.md",
    "model_cards/Boldt-Reranker-DE-350M-v1.md",
]

ADR_SECTIONS = (
    "## Status", "## Context", "## Decision", "## Alternatives",
    "## Consequences", "## Test/benchmark criteria",
)
MODEL_CARD_SECTIONS = (
    "## Intended use",
    "## Limitations",
    "## Evaluation",
    "## License",
    "## Reproducibility",
)

Issue = Tuple[str, str]


def check_structure() -> List[Issue]:
    issues: List[Issue] = []
    for rel in REQUIRED_DIRS:
        if not (ROOT / rel).is_dir():
            issues.append(("missing_dir", rel))
    for rel in REQUIRED_FILES:
        if not (ROOT / rel).exists():
            issues.append(("missing_file", rel))
    return issues


def check_json() -> List[Issue]:
    issues: List[Issue] = []
    for sub in ("configs", "benchmarks", "data/schema", "data/synthetic", "schemas"):
        for path in sorted((ROOT / sub).glob("*.json")):
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 - report any parse failure
                issues.append(("invalid_json", f"{path.relative_to(ROOT)}: {exc}"))
    return issues


def check_jsonl() -> List[Issue]:
    issues: List[Issue] = []
    for path in sorted((ROOT / "data" / "samples").glob("*.jsonl")) + sorted(
        (ROOT / "benchmarks").glob("*.jsonl")
    ):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except Exception as exc:  # noqa: BLE001
                issues.append(("invalid_jsonl", f"{path.relative_to(ROOT)}:{i}: {exc}"))
    return issues


def check_package_import() -> List[Issue]:
    issues: List[Issue] = []
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    try:
        import boldt_embed  # noqa: F401

        if not getattr(boldt_embed, "__version__", None):
            issues.append(("package_missing_version", "boldt_embed.__version__"))
    except Exception as exc:  # noqa: BLE001
        issues.append(("package_import_failed", str(exc)))
    return issues


def check_adrs() -> List[Issue]:
    issues: List[Issue] = []
    for path in sorted((ROOT / "docs" / "adr").glob("ADR-*.md")):
        text = path.read_text(encoding="utf-8")
        for section in ADR_SECTIONS:
            if section not in text:
                issues.append(("adr_missing_section", f"{path.name}: {section}"))
    return issues


def check_model_cards() -> List[Issue]:
    issues: List[Issue] = []
    for path in sorted((ROOT / "model_cards").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for section in MODEL_CARD_SECTIONS:
            if section not in text:
                issues.append(("model_card_missing_section", f"{path.name}: {section}"))
    return issues


_SCAN_SKIP_DIRS = {".git", "outputs", "node_modules", ".venv", "__pycache__"}


def check_placeholders() -> List[Issue]:
    """Scan authored .md for unrendered placeholders. Generated artifacts under outputs/
    (e.g. SentenceTransformers' auto model card, which contains Jinja {{ }}) are skipped."""
    issues: List[Issue] = []
    for path in ROOT.rglob("*.md"):
        if _SCAN_SKIP_DIRS.intersection(path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        if "{{" in text or "}}" in text:
            issues.append(("unrendered_placeholder", str(path.relative_to(ROOT))))
    return issues


def run_checks() -> Dict[str, object]:
    checks = {
        "structure": check_structure(),
        "json": check_json(),
        "jsonl": check_jsonl(),
        "package_import": check_package_import(),
        "adrs": check_adrs(),
        "model_cards": check_model_cards(),
        "placeholders": check_placeholders(),
    }
    issues = [i for group in checks.values() for i in group]
    return {"status": "pass" if not issues else "fail", "issue_count": len(issues), "checks": checks}


def render_markdown(report: Dict[str, object]) -> str:
    lines = [
        "# Boldt-Embed-DE Repo Validation Report",
        "",
        f"Status: **{report['status']}**",
        f"Issue count: {report['issue_count']}",
        "",
    ]
    for name, issues in report["checks"].items():  # type: ignore[union-attr]
        lines.append(f"## {name}")
        if not issues:
            lines.append("PASS")
        else:
            for kind, detail in issues:
                lines.append(f"- {kind}: {detail}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    args = parser.parse_args()
    report = run_checks()
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(report))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
