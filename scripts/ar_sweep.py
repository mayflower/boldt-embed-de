#!/usr/bin/env python3
"""Run N REAL AutoResearch training iterations over a leakage-safe knob grid (temperature / lr /
warmup), each via ``scripts/ar_loop.py --real --allow-gpu --allow-checkpoints``, scored vs the
baseline, logged to results.tsv, best tracked. Stdlib orchestrator — all ML runs inside ar_loop's
subprocess. The grid varies only OPTIMIZER/LOSS knobs and uses the base data mixture, so the
training data is fixed and the leakage gate (verified-clean prepared manifest) covers it once.

``--dry-run`` prints the planned grid and runs no training. A full sweep is long (hours); use
``--max-steps`` to pick a proxy budget, and ``--keep-only-best`` to bound disk to one checkpoint.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]

# leakage-SAFE knobs (now wired to real training); the data mixture stays at the base default.
TEMPS = [0.05, 0.03, 0.02]
LRS = [2e-5, 1e-5, 3e-5]
WARMUPS = [0.05, 0.0, 0.1]


def make_grid(n: int) -> List[Dict[str, Any]]:
    combos = list(itertools.product(TEMPS, LRS, WARMUPS))[:n]
    return [{"loop": i + 1, "run_id": f"sweep-{i + 1:02d}",
             "temperature": t, "learning_rate": lr, "warmup_ratio": w}
            for i, (t, lr, w) in enumerate(combos)]


# LEAKAGE-CLEAN training data (322 WebFAQ-overlapping rows removed; see leakage_report.json).
# The recipe trains on runtime.train_pairs, so we override the base default (raw, leaky) here —
# else the manifest's "clean" status would not match what is actually trained.
CLEAN_TRAIN_PAIRS = "outputs/autoresearch/prepared/rag_pairs_clean.jsonl"


def _load_json(p: pathlib.Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def config_for(g: Dict[str, Any], max_steps: int) -> Dict[str, Any]:
    """Per-loop experiment config: base mixture + the swept optimizer/loss knobs, on CLEAN data."""
    return {
        "extends": "configs/autoresearch/base_dense.json",
        "name": g["run_id"],
        "loss": {"temperature": g["temperature"]},
        # seq length capped at the v6.1-proven 256 (recipe uses max(query,doc)); document_length
        # 1024 @ batch 32 OOMs the A6000, and v6.1 itself trained at 256 (recall 0.9765).
        "training": {"learning_rate": g["learning_rate"], "warmup_ratio": g["warmup_ratio"],
                     "max_steps": max_steps, "max_query_length": 256, "max_document_length": 256,
                     "budget_minutes": 20},
        "runtime": {"dry_run": False, "allow_gpu": True, "write_checkpoints": True,
                    "train_pairs": CLEAN_TRAIN_PAIRS},
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loops", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=1000,
                    help="per-loop training steps (lower = faster proxy loops)")
    ap.add_argument("--baseline", default=str(ROOT / "outputs/autoresearch/baseline/metrics.json"))
    ap.add_argument("--prepared-manifest",
                    default=str(ROOT / "outputs/autoresearch/prepared/prepare_manifest.json"))
    ap.add_argument("--results", default=str(ROOT / "outputs/autoresearch/results.tsv"))
    ap.add_argument("--out-root", default=str(ROOT / "outputs/autoresearch/runs"))
    ap.add_argument("--work", default=str(ROOT / "outputs/autoresearch/sweep"))
    ap.add_argument("--keep-only-best", action="store_true",
                    help="delete non-best checkpoints during the run to bound disk")
    ap.add_argument("--dry-run", action="store_true", help="print the grid; run no training")
    args = ap.parse_args(argv)

    grid = make_grid(args.loops)
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(json.dumps({"loops": len(grid), "max_steps": args.max_steps,
                          "knobs": {"temperature": TEMPS, "learning_rate": LRS,
                                    "warmup_ratio": WARMUPS}, "grid": grid},
                         ensure_ascii=False, indent=2))
        return 0

    try:
        import torch  # noqa: F401
    except Exception:
        print("error: a real sweep needs torch — run under `conda run -n boldtembed`",
              file=sys.stderr)
        return 2

    def _ckpt(run_id: str) -> pathlib.Path:
        return pathlib.Path(args.out_root) / run_id / "checkpoint"

    rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for g in grid:
        cfg_path = work / f"{g['run_id']}.json"
        cfg_path.write_text(json.dumps(config_for(g, args.max_steps), ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"[sweep] loop {g['loop']}/{len(grid)} {g['run_id']}: "
              f"temp={g['temperature']} lr={g['learning_rate']} warmup={g['warmup_ratio']} "
              f"steps={args.max_steps}", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/ar_loop.py"), "--real", "--allow-gpu",
             "--allow-checkpoints", "--config", str(cfg_path), "--run-id", g["run_id"],
             "--out-root", args.out_root, "--baseline", args.baseline,
             "--prepared-manifest", args.prepared_manifest, "--results", args.results,
             "--status", "keep", "--notes", f"sweep loop {g['loop']}"],
            cwd=str(ROOT))

        run_dir = pathlib.Path(args.out_root) / g["run_id"]
        m = _load_json(run_dir / "metrics.json")
        s = _load_json(run_dir / "score.json")
        wf = (m.get("metrics", {}).get("webfaq") or {}).get("recall@100")
        promotable = m.get("status") in ("ok", "pass") and s.get("status") == "pass"
        row = {k: g[k] for k in ("loop", "run_id", "temperature", "learning_rate", "warmup_ratio")}
        row.update({"trial_status": m.get("status"), "score": s.get("score"),
                    "score_status": s.get("status"),
                    "failed_gates": [x.get("name") for x in s.get("failed_gates", [])],
                    "webfaq_recall100": wf, "promotable": promotable})
        rows.append(row)

        prev_best = best["run_id"] if best else None
        better = row["score"] is not None and (best is None or row["score"] > best["score"])
        if promotable and better:
            best = row
        if args.keep_only_best:                       # bound disk: keep only the current best ckpt
            keep = best["run_id"] if best else None
            for rid in {g["run_id"], prev_best}:
                if rid and rid != keep and _ckpt(rid).exists():
                    shutil.rmtree(_ckpt(rid), ignore_errors=True)
        (work / "sweep_summary.json").write_text(json.dumps(
            {"loops_run": len(rows), "promotable_count": sum(1 for r in rows if r["promotable"]),
             "best": best, "baseline": args.baseline, "rows": rows},
            ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"loops_run": len(rows),
                      "promotable": sum(1 for r in rows if r["promotable"]),
                      "best_run": best["run_id"] if best else None,
                      "best_webfaq_recall100": best["webfaq_recall100"] if best else None,
                      "summary": str(work / "sweep_summary.json")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
