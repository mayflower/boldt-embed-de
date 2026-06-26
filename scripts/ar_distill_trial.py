#!/usr/bin/env python3
"""Listwise-KL distillation trial (stdlib planner; real training via train_listwise_kl.py).

Validates the base checkpoint + teacher-scored lists (fail-closed, reusing the validator from
ar_prepare_listwise_distill), builds the train_listwise_kl.py command and the downstream MTEB eval
plan, and writes a manifest. --dry-run plans only; --real --allow-gpu --allow-checkpoints trains.

    python scripts/ar_distill_trial.py --config configs/autoresearch/distill/listwise_kl_v8.json --dry-run
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "ar_prepare_listwise_distill", ROOT / "scripts" / "ar_prepare_listwise_distill.py")
_prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prep)
validate_listwise_file = _prep.validate_listwise_file


def _base_ok(base: str) -> Optional[str]:
    if not base:
        return "base_checkpoint is required"
    looks_local = base.startswith(("./", "/", "outputs/", "data/", "checkpoints/", "models/", "runs/"))
    if looks_local and not (ROOT / base).exists() and not Path(base).exists():
        return f"base_checkpoint does not exist: {base}"
    return None


def plan_distill(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate + build the train + MTEB-eval command plan + manifest. Never executes."""
    errors: List[str] = []
    base = cfg.get("base_checkpoint", "")
    be = _base_ok(base)
    if be:
        errors.append(be)

    lists = cfg.get("lists", "")
    if not lists:
        # fail-closed with a named error instead of opening Path('') (the repo root, a directory)
        errors.append("config has no 'lists' path (listwise-KL needs teacher-scored candidate lists)")
        stats = {"path": None}
    else:
        lists_path = ROOT / lists if not Path(lists).is_absolute() else Path(lists)
        stats, list_errors = validate_listwise_file(lists_path)
        errors.extend(list_errors)

    t = cfg.get("training", {}) or {}
    output = cfg.get("output", "outputs/v8/distill/{}/checkpoint".format(cfg.get("name", "distill")))
    run_id = cfg.get("name", "v8-distill")
    train_cmd = [
        "python", "scripts/train_listwise_kl.py", "--base", base, "--lists", lists,
        "--output", output, "--steps", str(t.get("steps", 1500)),
        "--batch-queries", str(t.get("batch_queries", 4)), "--list-k", str(t.get("list_k", 24)),
        "--max-seq-length", str(t.get("max_seq_length", 256)), "--lr", str(t.get("lr", 5e-6)),
        "--tau", str(t.get("tau", 0.05)),
        "--contrastive-weight", str(t.get("contrastive_weight", 0.0)),
        "--run-id", run_id,
    ]
    label = run_id
    eval_plan = [
        "CUDA_VISIBLE_DEVICES=0 python scripts/run_mteb_retrieval_de.py "
        f"--model {output} --label {label} "
        "--tasks GermanQuAD-Retrieval,GerDaLIRSmall,MIRACLRetrievalHardNegatives,MultiLongDocRetrieval "
        "--loader st --batch-size 32 --max-seq-length 512",
        f"python scripts/ar_promote.py --candidate {label} --format markdown",
    ]
    manifest = {
        "name": run_id, "base_checkpoint": base, "lists": lists,
        "list_hash": stats.get("list_hash"), "list_validation": stats,
        "training": t, "output_checkpoint": output, "eval_plan": eval_plan,
    }
    return {"errors": errors, "run_id": run_id, "output": output,
            "command": " ".join(train_cmd), "command_argv": train_cmd,
            "eval_plan": eval_plan, "manifest": manifest}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="outputs/autoresearch/distill")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--allow-gpu", action="store_true")
    ap.add_argument("--allow-checkpoints", action="store_true")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    plan = plan_distill(cfg)

    out_dir = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{plan['run_id']}_distill_manifest.json").write_text(
        json.dumps(plan["manifest"], ensure_ascii=False, indent=2), encoding="utf-8")

    if plan["errors"]:
        print(json.dumps({"status": "fail", "errors": plan["errors"]}, ensure_ascii=False, indent=2))
        return 1
    if not (args.real and args.allow_gpu and args.allow_checkpoints):
        print(json.dumps({"status": "dry-run", "run_id": plan["run_id"], "command": plan["command"],
                          "output": plan["output"], "eval_plan": plan["eval_plan"],
                          "note": "pass --real --allow-gpu --allow-checkpoints to train"},
                         ensure_ascii=False, indent=2))
        return 0
    proc = subprocess.run([sys.executable] + plan["command_argv"][1:], cwd=str(ROOT))
    print(json.dumps({"status": "ok" if proc.returncode == 0 else "fail",
                      "run_id": plan["run_id"], "output": plan["output"],
                      "returncode": proc.returncode, "next": plan["eval_plan"]},
                     ensure_ascii=False, indent=2))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
