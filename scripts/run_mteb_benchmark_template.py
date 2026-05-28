#!/usr/bin/env python3
"""Scaffold for REAL MTEB evaluation of a trained Boldt embedder.

Requires the eval extras and a trained, SentenceTransformers-compatible model:
    pip install -e '.[eval]'

It is intentionally not part of the stdlib smoke gates (needs weights + downloads).
Per ADR-005, results are written with run metadata so every reported number is auditable.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="SentenceTransformers path or HF id")
    parser.add_argument("--config", default="benchmarks/mteb_german_tasks.json")
    parser.add_argument("--output-dir", default="outputs/mteb")
    parser.add_argument("--task", action="append", help="Override task name; repeatable")
    args = parser.parse_args()

    try:
        import mteb
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - requires extras
        raise SystemExit(
            f"Missing eval extras. Install: pip install -e '.[eval]'. Error: {exc}"
        )

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    task_names = args.task or [t["name"] for t in config["suggested_tasks"]]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Run metadata (ADR-005): a number without this is treated as not-reported.
    metadata = {
        "command": "run_mteb_benchmark_template.py",
        "commit": _git_commit(),
        "model": args.model,
        "tasks": task_names,
        "languages": config.get("languages"),
        "hardware": platform.platform(),
        "output_path": str(out),
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    model = SentenceTransformer(args.model)  # pragma: no cover
    tasks = mteb.get_tasks(tasks=task_names)  # pragma: no cover
    results = mteb.evaluate(model, tasks, output_folder=str(out))  # pragma: no cover
    print(json.dumps(metadata, indent=2))
    print(results)  # pragma: no cover
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
