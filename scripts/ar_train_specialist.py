#!/usr/bin/env python3
"""Train ONE domain specialist from a shared warm-start (stdlib planner; real run via ar_loop.py).

Specialists are domain-specific checkpoints ALL warm-started from the same basin, so a later
model-soup/SLERP/TIES merge is meaningful (Prompt 07). This script validates the source against the
catalogue (fail-closed: training_usable + leakage-clean) and the warm-start, then either:
  --dry-run (default): writes the specialist experiment config + plan + manifest + the exact
    ar_loop.py command it WOULD run — no GPU, no training.
  --real --allow-gpu --allow-checkpoints: invokes ar_loop.py to materialize a single-source clean
    mixture and train to outputs/<out-root>/spec-<label>/checkpoint.

    python scripts/ar_train_specialist.py --config configs/autoresearch/specialists/v8_specialists.json \
        --source-id swim_ir_de_full --out-root outputs/v8/specialists --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import autoresearch_recipe as recipe  # noqa: E402  (stdlib; _load_catalogue)

BASE_CONFIG = "configs/autoresearch/base_dense.json"


def _source_hash(path: Path) -> str:
    """Content fingerprint: sha256 over file size + the first/last 64 KiB (cheap on huge corpora,
    deterministic for tests on small files)."""
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with path.open("rb") as fh:
        h.update(fh.read(65536))
        if size > 131072:
            fh.seek(-65536, 2)
            h.update(fh.read(65536))
    return h.hexdigest()


def _warm_start_ok(warm_start: str) -> Optional[str]:
    """None if the warm-start is usable, else an error string. Local paths must exist; a non-local
    ref (e.g. an HF id) is allowed (resolved at real-run time)."""
    if not warm_start:
        return "warm_start is required (specialists must share a basin)"
    looks_local = warm_start.startswith(("./", "/", "outputs/", "data/", "checkpoints/", "models/", "runs/"))
    if looks_local and not (ROOT / warm_start).exists() and not Path(warm_start).exists():
        return f"warm_start path does not exist: {warm_start}"
    return None


def plan_specialist(spec_config: Dict[str, Any], source_id: str, out_root: str,
                    catalogue: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Pure planner: validate + build the specialist experiment config, manifest and ar_loop command.
    Returns a dict with keys: errors[], label, experiment_config, command, manifest. Never executes."""
    catalogue = catalogue if catalogue is not None else recipe._load_catalogue()
    errors: List[str] = []

    sources = {s["id"]: s for s in spec_config.get("sources", []) if isinstance(s, dict) and s.get("id")}
    src = sources.get(source_id)
    if src is None:
        errors.append(f"source-id {source_id!r} not listed in this specialists config "
                      f"(have: {sorted(sources)})")
    label = (src or {}).get("label", source_id)
    steps = int((src or {}).get("steps", spec_config.get("default_steps", 6000)))
    warm_start = spec_config.get("warm_start", "")

    rec = catalogue.get(source_id)
    if rec is None:
        errors.append(f"source {source_id!r} is not in configs/data_sources.json")
    else:
        if not rec.get("training_usable"):
            errors.append(f"source {source_id!r} is training_usable=false")
        if rec.get("leakage") not in ("scanned_clean", "clean"):
            errors.append(f"source {source_id!r} leakage={rec.get('leakage')!r} — only "
                          "scanned_clean/clean sources may train a specialist")
    ws_err = _warm_start_ok(warm_start)
    if ws_err:
        errors.append(ws_err)

    rows = 0
    source_hash = None
    if rec is not None:
        rows = int(rec.get("rows") or rec.get("rows_clean") or 0)
        spath = rec.get("path")
        if spath and (ROOT / spath).exists():
            source_hash = _source_hash(ROOT / spath)

    training = dict(spec_config.get("training", {}) or {})
    training["max_steps"] = steps
    run_id = f"spec-{label}"
    ckpt = f"{out_root.rstrip('/')}/{run_id}/checkpoint"
    # The v6.1 trainer needs rank-promotion triplets (hard negatives) — a materialized single-source
    # mixture has none. So a specialist needs a hard_negatives file (mine it with /ar-refresh-hardnegs
    # on the mixture first). Surface its absence at PLAN time so a real run doesn't fail deep in the
    # recipe input gate; if a local hard_negatives path is given but missing, that's a hard error.
    hard_negatives = (src or {}).get("hard_negatives") or spec_config.get("hard_negatives")
    warnings: List[str] = []
    if hard_negatives:
        hn = Path(hard_negatives)
        if not hn.is_absolute() and not (ROOT / hard_negatives).exists() and not hn.exists():
            errors.append(f"hard_negatives path does not exist: {hard_negatives}")
    else:
        warnings.append("no hard_negatives configured — a REAL run will fail the recipe input gate "
                        "(v6.1 training needs rank-promotion triplets). Mine them with "
                        "/ar-refresh-hardnegs on this source's mixture, then set `hard_negatives`.")
    runtime = {
        "dry_run": False, "allow_gpu": True, "write_checkpoints": True, "train": True,
        "materialize_mixture": True, "mixture_total": rows or 100000,
        "train_base_model": warm_start,
    }
    if hard_negatives:
        runtime["hard_negatives"] = hard_negatives
    experiment_config = {
        "extends": BASE_CONFIG,
        "name": run_id,
        "data_mixture": {source_id: 1.0},
        "training": training,
        "runtime": runtime,
    }
    command = [
        "python", "scripts/ar_loop.py", "--config", f"{out_root.rstrip('/')}/{run_id}/experiment.json",
        "--real", "--allow-gpu", "--allow-checkpoints",
        "--out-root", out_root, "--run-id", run_id, "--status", "keep",
        "--baseline", "outputs/autoresearch/baseline/metrics.json",
    ]
    manifest = {
        "name": run_id, "label": label, "source_id": source_id, "source_hash": source_hash,
        "source_rows": rows, "warm_start": warm_start, "steps": steps,
        "training": training, "out_root": out_root, "checkpoint": ckpt,
        "hard_negatives": hard_negatives,
        "leakage": {"status": (rec or {}).get("leakage"), "basis": "source_catalogue"},
    }
    return {"errors": errors, "warnings": warnings, "label": label, "run_id": run_id,
            "hard_negatives": hard_negatives, "experiment_config": experiment_config,
            "command": " ".join(command), "command_argv": command, "manifest": manifest,
            "checkpoint": ckpt}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--out-root", default="outputs/v8/specialists")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--allow-gpu", action="store_true")
    ap.add_argument("--allow-checkpoints", action="store_true")
    args = ap.parse_args(argv)

    spec_config = json.loads((ROOT / args.config).read_text(encoding="utf-8")
                             if not Path(args.config).is_absolute()
                             else Path(args.config).read_text(encoding="utf-8"))
    plan = plan_specialist(spec_config, args.source_id, args.out_root)

    run_dir = ROOT / args.out_root / plan["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment.json").write_text(
        json.dumps(plan["experiment_config"], ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "specialist_manifest.json").write_text(
        json.dumps(plan["manifest"], ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "plan.json").write_text(
        json.dumps({k: plan[k] for k in ("run_id", "label", "command", "errors", "warnings",
                                          "hard_negatives", "checkpoint")},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    if plan["errors"]:
        print(json.dumps({"status": "fail", "errors": plan["errors"]}, ensure_ascii=False, indent=2))
        return 1

    if not (args.real and args.allow_gpu and args.allow_checkpoints):
        print(json.dumps({"status": "dry-run", "run_id": plan["run_id"], "label": plan["label"],
                          "command": plan["command"], "checkpoint": plan["checkpoint"],
                          "warnings": plan["warnings"],
                          "note": "plan + experiment.json + manifest written; pass --real --allow-gpu "
                                  "--allow-checkpoints to train"},
                         ensure_ascii=False, indent=2))
        return 0

    # real: a v6.1 specialist needs hard negatives; refuse fast (don't fail deep in the recipe).
    if not plan["hard_negatives"]:
        print(json.dumps({"status": "fail", "errors": plan["warnings"],
                          "hint": "run /ar-refresh-hardnegs on this source, then set hard_negatives "
                                  "in the specialists config"}, ensure_ascii=False, indent=2))
        return 1

    # real: hand off to ar_loop (which materializes the single-source clean mix and trains)
    proc = subprocess.run([sys.executable] + plan["command_argv"][1:], cwd=str(ROOT))
    print(json.dumps({"status": "ok" if proc.returncode == 0 else "fail",
                      "run_id": plan["run_id"], "checkpoint": plan["checkpoint"],
                      "ar_loop_returncode": proc.returncode}, ensure_ascii=False, indent=2))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
