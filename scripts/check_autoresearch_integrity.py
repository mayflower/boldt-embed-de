#!/usr/bin/env python3
"""Guard the AutoResearch protected surfaces.

The AutoResearch loop may edit ONLY the experiment configs and the dense recipe. Everything that
defines how a trial is judged — scoring, gates, release validation, eval datasets, leakage checks,
benchmark harnesses, and baseline outputs — is protected. This script classifies the changed paths
(from ``git status`` or an explicit ``--paths`` list) and **fails** if any protected surface was
touched.

Editable surface:
    configs/autoresearch/experiments/*.json
    src/boldt_embed/autoresearch_recipe.py

Use ``--strict`` to additionally fail when anything OTHER than the editable surface changed.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

EDITABLE_GLOBS = [
    "configs/autoresearch/experiments/*.json",
    "src/boldt_embed/autoresearch_recipe.py",
]

PROTECTED_GLOBS = [
    # scoring / gates / release validation / loop infrastructure
    "scripts/ar_score.py",
    "scripts/ar_prepare.py",
    "scripts/ar_run_trial.py",
    "scripts/ar_log_result.py",
    "scripts/check_autoresearch_integrity.py",
    "scripts/validate_release_2026.py",
    "scripts/check_*.py",
    # evaluation harnesses
    "scripts/eval_*.py",
    "src/boldt_embed/eval_harness.py",
    "src/boldt_embed/metrics.py",
    # leakage checks
    "src/boldt_embed/leakage_index.py",
    "scripts/*leakage*.py",
    # evaluation data / manifests / baselines / benchmarks
    "data/processed/eval/**",
    "outputs/v4-rag-reranker/eval/**",
    "**/eval_manifest.json",
    "outputs/autoresearch/baseline/**",
    "benchmarks/**",
    # the base config drives every trial via `extends` (data mix, train paths, thresholds);
    # only the per-experiment overlays in experiments/*.json are loop-editable.
    "configs/autoresearch/base_*.json",
]


def _glob_to_re(glob: str) -> re.Pattern:
    out = []
    i, n = 0, len(glob)
    while i < n:
        c = glob[i]
        if glob.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif glob.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(out) + "$")


_EDITABLE_RE = [_glob_to_re(g) for g in EDITABLE_GLOBS]
_PROTECTED_RE = [_glob_to_re(g) for g in PROTECTED_GLOBS]


def _norm(path: str) -> str:
    return path.strip().lstrip("./").replace("\\", "/")


def classify_paths(paths: List[str]) -> Dict[str, List[str]]:
    """Bucket each changed path into editable / protected / other. Pure function."""
    editable, protected, other = [], [], []
    for raw in paths:
        p = _norm(raw)
        if not p:
            continue
        if any(rx.match(p) for rx in _EDITABLE_RE):
            editable.append(p)
        elif any(rx.match(p) for rx in _PROTECTED_RE):
            protected.append(p)
        else:
            other.append(p)
    return {"editable": sorted(set(editable)), "protected": sorted(set(protected)),
            "other": sorted(set(other))}


def _porcelain_paths() -> List[str]:
    """Uncommitted working-tree changes (modified + staged + untracked)."""
    paths: List[str] = []
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=15)
    except Exception:
        return paths
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        body = line[3:]
        if " -> " in body:  # rename: "old -> new"
            old, new = body.split(" -> ", 1)
            paths.extend([old.strip(), new.strip()])
        else:
            paths.append(body.strip())
    return paths


def _changed_paths(base_ref: Optional[str] = None) -> List[str]:
    """Changed paths to vet. Always includes uncommitted working-tree changes; when ``base_ref``
    is given, ALSO includes everything committed since that ref — so a loop that edits a protected
    file and commits it cannot escape the gate. Pass the loop's start commit as ``base_ref``."""
    paths = _porcelain_paths()
    if base_ref:
        try:
            out = subprocess.run(["git", "diff", "--name-only", base_ref], cwd=str(ROOT),
                                 capture_output=True, text=True, timeout=20)
            if out.returncode == 0:
                paths += [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        except Exception:
            pass
    return paths


def evaluate(paths: List[str], strict: bool = False) -> Dict[str, object]:
    cls = classify_paths(paths)
    violations = list(cls["protected"])
    if strict:
        violations += cls["other"]
    status = "pass" if not violations else "fail"
    return {"status": status, "violations": sorted(set(violations)), **cls}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", nargs="*", default=None,
                    help="explicit changed paths to check (default: read from git)")
    ap.add_argument("--base-ref", default=None,
                    help="also vet everything committed since this ref (e.g. the loop's start "
                         "commit) so committing a protected-surface edit cannot bypass the gate")
    ap.add_argument("--strict", action="store_true",
                    help="also fail if anything other than the editable surface changed")
    ap.add_argument("--format", choices=["json", "text"], default="text")
    args = ap.parse_args(argv)

    paths = args.paths if args.paths is not None else _changed_paths(args.base_ref)
    result = evaluate(paths, strict=args.strict)

    if args.format == "json":
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"AutoResearch integrity: {result['status'].upper()}")
        if result["editable"]:
            print("  editable touched: " + ", ".join(result["editable"]))
        if result["violations"]:
            print("  PROTECTED surfaces touched (not allowed):")
            for v in result["violations"]:
                print(f"    ✗ {v}")
        elif not paths:
            print("  (no changed paths detected)")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
