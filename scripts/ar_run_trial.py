#!/usr/bin/env python3
"""Safe 20-minute trial runner for dense-retriever AutoResearch.

Enforces the **20-minute default budget**: a budget over 20 fails unless ``--allow-longer-than-20``
is passed, and such a run is stamped ``invalid_for_default_loop: true``. Resolves ``extends`` config
inheritance, computes a monotonic deadline, calls
``boldt_embed.autoresearch_recipe.run_dense_trial`` (dry-run by default), and always writes an
auditable run directory (``config.resolved.json``, ``command.txt``, ``env.json``, ``git.diffstat``,
``git.status``, ``metrics.json``, ``run_card.md``). A crash is captured into ``error.json`` plus a
failed ``metrics.json`` rather than leaving an empty directory. Real successful trials also emit a
canonical run card via ``experiment_registry`` so the repo's provenance tooling sees them.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import experiment_registry as registry  # noqa: E402  (stdlib provenance helpers)

DEFAULT_BUDGET_MINUTES = 20


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_config(config_path: pathlib.Path) -> Dict[str, Any]:
    """Load a config, merging a base referenced via ``extends`` (relative to repo root)."""
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(cfg.get("extends"), str):
        base = json.loads((ROOT / cfg["extends"]).read_text(encoding="utf-8"))
        merged = _deep_merge(base, {k: v for k, v in cfg.items() if k != "extends"})
        merged["_extends"] = cfg["extends"]
        return merged
    return cfg


def _git(cmd: List[str]) -> str:
    try:
        out = subprocess.run(["git"] + cmd, cwd=str(ROOT), capture_output=True,
                             text=True, timeout=15)
        return out.stdout if out.returncode == 0 else ""
    except Exception:
        return ""


def _env_info() -> Dict[str, Any]:
    """Reuse the repo's canonical env capture (experiment_registry) so AutoResearch provenance
    does not drift from the rest of the project; add the runtime-specific bits."""
    env = registry.collect_env_metadata()
    env["conda_env"] = os.environ.get("CONDA_DEFAULT_ENV")
    env["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    env["accelerate"] = registry._pkg_version("accelerate")
    env["datasets"] = registry._pkg_version("datasets")
    return env


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-minutes", type=int, default=DEFAULT_BUDGET_MINUTES)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--allow-gpu", action="store_true")
    ap.add_argument("--allow-checkpoints", action="store_true")
    ap.add_argument("--allow-longer-than-20", action="store_true")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--prepared-manifest", default=None)
    ap.add_argument("--notes", default=None)
    args = ap.parse_args(argv)

    if args.real and args.dry_run:
        print("error: pass either --dry-run or --real, not both", file=sys.stderr)
        return 2
    dry_run = not args.real

    # --- budget enforcement (the central safety rule) ---
    budget = args.budget_minutes
    invalid_for_default_loop = False
    if budget > DEFAULT_BUDGET_MINUTES:
        if not args.allow_longer_than_20:
            print(f"error: --budget-minutes {budget} exceeds the 20-minute default; pass "
                  f"--allow-longer-than-20 to override", file=sys.stderr)
            return 2
        invalid_for_default_loop = True

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = resolve_config(pathlib.Path(args.config))
    runtime = dict(cfg.get("runtime", {}) or {})
    if args.seed is not None:
        cfg["seed"] = args.seed
    runtime["dry_run"] = dry_run
    runtime["allow_gpu"] = bool(args.allow_gpu)
    # Training writes a checkpoint, so only enable it when checkpoints are explicitly allowed.
    runtime["write_checkpoints"] = bool(args.allow_checkpoints)
    if not dry_run:
        runtime["train"] = bool(args.allow_checkpoints)
    # The recipe reads leakage status from the prepared manifest (never fabricates it).
    runtime["prepared_manifest"] = args.prepared_manifest
    cfg["runtime"] = runtime
    cfg["budget_minutes"] = budget
    cfg["run_id"] = out_dir.name

    # --real needs an explicit hardware opt-in; no config field may override this human gate.
    if not dry_run and not args.allow_gpu:
        print("error: --real requires --allow-gpu", file=sys.stderr)
        return 2

    # --- persist inputs/provenance ---
    (out_dir / "config.resolved.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "command.txt").write_text(
        "python " + " ".join([sys.argv[0]] + (argv if argv is not None else sys.argv[1:])) + "\n",
        encoding="utf-8")
    (out_dir / "env.json").write_text(
        json.dumps(_env_info(), ensure_ascii=False, indent=2), encoding="utf-8")
    commit = registry.current_git_commit()
    commit = None if commit in ("", "unknown") else commit
    status_short = _git(["status", "--short"])
    (out_dir / "git.status").write_text(status_short, encoding="utf-8")
    # diffstat only — a full working-tree diff can be unbounded; commit + status fix provenance.
    (out_dir / "git.diffstat").write_text(_git(["diff", "--stat"]), encoding="utf-8")
    git_info = {"commit": commit, "dirty": bool(status_short.strip())}

    from boldt_embed import autoresearch_recipe as recipe  # stdlib import; ML is lazy/subprocess

    deadline_epoch_s = time.monotonic() + budget * 60
    t0 = time.monotonic()
    try:
        result = recipe.run_dense_trial(cfg, out_dir, deadline_epoch_s, dry_run=dry_run)
        crashed = False
    except Exception as exc:  # never leave an empty run dir
        crashed = True
        tb = traceback.format_exc()
        (out_dir / "error.json").write_text(
            json.dumps({"error": str(exc), "type": type(exc).__name__, "traceback": tb},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        result = {"run_id": out_dir.name, "status": "crash",
                  "metrics": recipe._metrics_skeleton(out_dir.name, "crash")["metrics"],
                  "note": f"recipe raised {type(exc).__name__}: {exc}"}
    elapsed = round(time.monotonic() - t0, 3)

    deadline_respected = (elapsed <= budget * 60 + 1.0) and result.get("deadline_respected", True)
    metrics_doc: Dict[str, Any] = {
        "run_id": result.get("run_id", out_dir.name),
        "status": "crash" if crashed else result.get("status", "ok"),
        "mode": "dry_run" if dry_run else "real",
        "budget_minutes": budget,
        "elapsed_seconds": elapsed,
        "deadline_respected": bool(deadline_respected),
        "invalid_for_default_loop": invalid_for_default_loop,
        "config_path": args.config,
        "baseline_path": args.baseline,
        "prepared_manifest": args.prepared_manifest,
        "git": git_info,
        "metrics": result.get("metrics", {}),
    }
    for k in ("scale_disclaimer", "note", "missing_inputs", "training_plan", "eval_model",
              "leakage_status", "trained"):
        if k in result:
            metrics_doc[k] = result[k]
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    # Emit a canonical run card (experiment_registry) for REAL successful trials, so the repo's
    # provenance/summary tooling sees AutoResearch runs. Dry-runs emit nothing (plumbing only).
    if not dry_run and not crashed and metrics_doc["status"] in ("ok", "pass"):
        try:
            registry.emit_run_card(
                run_id=f"autoresearch-{metrics_doc['run_id']}",
                run_type="train_embedder" if result.get("trained") else "eval",
                command=(out_dir / "command.txt").read_text(encoding="utf-8").strip(),
                model=result.get("eval_model"),
                dataset=str(runtime.get("eval_sets", "")),
                metrics=metrics_doc["metrics"],
                input_artifacts=[args.config] + ([args.prepared_manifest]
                                                 if args.prepared_manifest else []),
                output_artifacts=[str(out_dir / "metrics.json")],
                notes=f"AutoResearch real trial; leakage={result.get('leakage_status')}")
        except Exception:  # provenance is best-effort; never fail a trial on run-card emission
            pass

    _write_run_card(out_dir, metrics_doc, args, commit, git_info)

    print(json.dumps({"status": metrics_doc["status"], "mode": metrics_doc["mode"],
                      "elapsed_seconds": elapsed, "budget_minutes": budget,
                      "invalid_for_default_loop": invalid_for_default_loop,
                      "out": str(out_dir)}, ensure_ascii=False))
    return 0 if metrics_doc["status"] in ("ok", "pass") else 1


def _fmt(metrics: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = metrics
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _write_run_card(out_dir: pathlib.Path, doc: Dict[str, Any], args: argparse.Namespace,
                    commit: Optional[str], git_info: Dict[str, Any]) -> None:
    m = doc.get("metrics", {})
    rows = [
        ("WebFAQ Recall@100", _fmt(m, ["webfaq", "recall@100"])),
        ("WebFAQ nDCG@10", _fmt(m, ["webfaq", "ndcg@10"])),
        ("GermanQuAD nDCG@10", _fmt(m, ["germanquad", "ndcg@10"])),
        ("DT-test nDCG@10", _fmt(m, ["dt_test", "ndcg@10"])),
        ("Matryoshka 256 retention", _fmt(m, ["matryoshka", "retention_256"])),
    ]
    lines = [
        f"# AutoResearch Run: {doc['run_id']}", "",
        f"Status: {doc['status']}",
        f"Mode: {doc['mode']}",
        f"Budget: {doc['budget_minutes']} minutes",
        f"Elapsed: {doc['elapsed_seconds']} seconds",
        f"Deadline respected: {'yes' if doc['deadline_respected'] else 'no'}",
        f"Invalid for default loop: {'yes' if doc['invalid_for_default_loop'] else 'no'}", "",
        "## Command", "", "```bash", (out_dir / 'command.txt').read_text(encoding='utf-8').strip(),
        "```", "",
        "## Git", "", f"- commit: {commit}", f"- dirty: {git_info['dirty']}",
        "- diffstat saved: git.diffstat", "",
        "## Config", "", f"- config path: {args.config}",
        "- resolved config: config.resolved.json", "",
        "## Metrics", "", "| metric | value |", "|---|---:|",
    ]
    lines += [f"| {name} | {val} |" for name, val in rows]
    if doc.get("scale_disclaimer"):
        lines += ["", f"> {doc['scale_disclaimer']}"]
    if doc.get("note"):
        lines += ["", f"Note: {doc['note']}"]
    if doc.get("missing_inputs"):
        lines += ["", "## Missing inputs", ""] + [f"- {x}" for x in doc["missing_inputs"]]
    lines += ["", "## Notes", "", (args.notes or "—")]
    (out_dir / "run_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
