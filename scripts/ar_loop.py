#!/usr/bin/env python3
"""Run ONE end-to-end AutoResearch iteration — designed to be driven from the Claude Code CLI.

Pipeline (reusing the canonical scripts, no logic duplicated): trial -> score -> log -> integrity.
Prints a single machine-readable JSON verdict so an operator/agent can read the outcome and decide
the next move. The OUTER loop — edit ``configs/autoresearch/experiments/current.json``, re-run —
is the agent's job; this command is one deterministic, auditable step.

Run it from Claude Code:

    # dry-run (stdlib only, no GPU): plumbing check
    conda run -n boldtembed python scripts/ar_loop.py --dry-run

    # real iteration on the RTX A6000 (eval-only of the configured model)
    conda run -n boldtembed python scripts/ar_loop.py --real --allow-gpu \
        --baseline outputs/autoresearch/baseline/metrics.json \
        --prepared-manifest outputs/autoresearch/prepared/prepare_manifest.json

    # real training trial (writes its checkpoint into the run dir, never the promoted path)
    conda run -n boldtembed python scripts/ar_loop.py --real --allow-gpu --allow-checkpoints ...

Exit code is 0 only when the trial ran, the score gate passed (if a baseline was given), and the
integrity check passed — so it is safe to use directly as a CLI gate.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import importlib.util
import io
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")


def _quiet(fn, argv: List[str]) -> int:
    """Call a script main(argv), swallowing its stdout so only our verdict is emitted."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(argv)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/autoresearch/experiments/current.json"))
    ap.add_argument("--out-root", default=str(ROOT / "outputs/autoresearch/runs"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--baseline", default=str(ROOT / "outputs/autoresearch/baseline/metrics.json"))
    ap.add_argument("--results", default=str(ROOT / "outputs/autoresearch/results.tsv"))
    ap.add_argument("--budget-minutes", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--allow-gpu", action="store_true")
    ap.add_argument("--allow-checkpoints", action="store_true")
    ap.add_argument("--allow-longer-than-20", action="store_true")
    ap.add_argument("--prepared-manifest", default=None)
    ap.add_argument("--base-ref", default=None,
                    help="loop start commit; passed to the integrity check so committed protected "
                         "edits are caught")
    ap.add_argument("--status", default="keep", help="results.tsv disposition for this row")
    ap.add_argument("--notes", default="autoresearch loop iteration")
    args = ap.parse_args(argv)

    if args.real and args.dry_run:
        print("error: pass either --dry-run or --real, not both", file=sys.stderr)
        return 2
    dry = not args.real

    # Fail fast with an actionable hint if a real run is launched without the project env.
    if args.real:
        try:
            import torch  # noqa: F401
        except Exception:
            print("error: --real needs torch; launch under the project env, e.g.:\n"
                  "  conda run -n boldtembed python scripts/ar_loop.py --real --allow-gpu ...",
                  file=sys.stderr)
            return 2

    run_id = args.run_id or f"ar-{'dry' if dry else 'real'}-{_stamp()}"
    out_dir = pathlib.Path(args.out_root) / run_id

    runner = _load("ar_run_trial")
    scorer = _load("ar_score")
    logger = _load("ar_log_result")
    integ = _load("check_autoresearch_integrity")

    # 1) trial -------------------------------------------------------------------------------
    trial_argv = ["--config", args.config, "--out", str(out_dir),
                  "--budget-minutes", str(args.budget_minutes), "--notes", args.notes]
    trial_argv += ["--dry-run"] if dry else ["--real"]
    if args.allow_gpu:
        trial_argv.append("--allow-gpu")
    if args.allow_checkpoints:
        trial_argv.append("--allow-checkpoints")
    if args.allow_longer_than_20:
        trial_argv.append("--allow-longer-than-20")
    if args.baseline:
        trial_argv += ["--baseline", args.baseline]
    if args.prepared_manifest:
        trial_argv += ["--prepared-manifest", args.prepared_manifest]
    trial_rc = _quiet(runner.main, trial_argv)

    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        print(json.dumps({"run_id": run_id, "stage": "trial", "trial_rc": trial_rc,
                          "error": "no metrics.json — budget guard rejected the run or it failed "
                                   "before writing outputs"}, ensure_ascii=False, indent=2))
        return trial_rc or 1
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    # 2) score (only if a real baseline file exists) -----------------------------------------
    score_doc: Optional[Dict[str, Any]] = None
    if args.baseline and pathlib.Path(args.baseline).exists():
        _quiet(scorer.main, ["--run", str(metrics_path), "--baseline", args.baseline,
                             "--out", str(out_dir / "score.json")])
        score_doc = json.loads((out_dir / "score.json").read_text(encoding="utf-8"))

    # 3) log one auditable row ---------------------------------------------------------------
    _quiet(logger.main, ["--run", str(out_dir), "--results", args.results,
                         "--status", args.status, "--notes", args.notes])

    # 4) integrity (no stdout; we read the structured result directly) -----------------------
    integ_result = integ.evaluate(integ._changed_paths(args.base_ref))

    verdict = {
        "run_id": run_id,
        "out": str(out_dir),
        "mode": metrics.get("mode"),
        "trial_status": metrics.get("status"),
        "score_status": (score_doc or {}).get("status"),
        "score": (score_doc or {}).get("score"),
        "failed_gates": [g["name"] for g in (score_doc or {}).get("failed_gates", [])],
        "leakage_status": (metrics.get("metrics", {}).get("leakage") or {}).get("status"),
        "integrity": integ_result["status"],
        "integrity_violations": integ_result["violations"],
        "results_tsv": args.results,
        "note": metrics.get("note"),
        "promotable": False,
    }
    promotable = (metrics.get("status") in ("ok", "pass")
                  and integ_result["status"] == "pass"
                  and score_doc is not None and score_doc.get("status") == "pass")
    verdict["promotable"] = promotable
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0 if promotable else 1


if __name__ == "__main__":
    raise SystemExit(main())
