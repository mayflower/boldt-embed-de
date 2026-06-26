#!/usr/bin/env python3
"""STUB (dry-run only) — plan a hybrid / late-interaction EVALUATION. No GPU, fail-closed on inputs.

Optional research track (docs/research/hybrid-multivector-ceiling-breaker.md). Plans which retrieval
modes to score on which tasks for a candidate; this is NOT the dense frontier gate and must never be
used to bless a hybrid system as the dense product.

    python scripts/ar_hybrid_eval.py --model outputs/v8/x/checkpoint --mode reranked_two_stage --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["GermanQuAD-Retrieval", "GerDaLIRSmall", "MIRACLRetrievalHardNegatives", "MultiLongDocRetrieval"]


def plan_eval(model: str, mode: str, tasks: List[str]) -> Dict[str, Any]:
    errors: List[str] = []
    if not model:
        errors.append("model is required")
    elif model.startswith(("./", "/", "outputs/", "data/")) and not (ROOT / model).exists() \
            and not Path(model).exists():
        errors.append(f"model path does not exist: {model}")
    return {"errors": errors, "mode": mode, "model": model, "tasks": tasks,
            "note": "dry-run plan only — separate hybrid gate, NOT the dense frontier gate"}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="reranked_two_stage")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    plan = plan_eval(args.model, args.mode, [t.strip() for t in args.tasks.split(",") if t.strip()])
    print(json.dumps({"status": "fail" if plan["errors"] else "dry-run", **plan},
                     ensure_ascii=False, indent=2))
    return 1 if plan["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
