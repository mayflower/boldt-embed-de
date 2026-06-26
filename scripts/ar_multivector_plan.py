#!/usr/bin/env python3
"""STUB (dry-run only) — plan a hybrid/multi-vector ceiling-breaker mode. No training, no GPU.

Optional research track (docs/research/hybrid-multivector-ceiling-breaker.md). Emits the steps +
artifacts a chosen mode would need so the controller can *plan* it without it ever passing as a dense
result. Promoting anything from this track needs its OWN gate — never the dense frontier gate.

    python scripts/ar_multivector_plan.py --mode colbert_late_interaction --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
TRACK_CONFIG = ROOT / "configs" / "autoresearch" / "hybrid_track.json"

_STEPS = {
    "sparse_dense_hybrid": ["build BM25/SPLADE sparse index", "fuse sparse+dense (RRF/weighted)",
                            "tune fusion weight on a dev split", "eval hybrid mode (separate gate)"],
    "splade_head": ["add learned sparse head", "train sparse objective on clean pairs",
                    "eval sparse + fused (separate gate)"],
    "bge_m3_style": ["enable dense+sparse+multivector heads", "joint train", "eval each signal"],
    "colbert_late_interaction": ["emit per-token vectors", "build MaxSim index",
                                 "eval late-interaction mode (separate gate)"],
    "reranked_two_stage": ["dense first-stage retrieve top-k", "rerank with Boldt reranker",
                           "eval two-stage nDCG (separate gate)"],
}


def plan_mode(mode: str, track: Dict[str, Any]) -> Dict[str, Any]:
    known = {m["name"] for m in track.get("modes", [])}
    if mode not in known:
        return {"errors": [f"unknown mode {mode!r}; known: {sorted(known)}"]}
    return {"errors": [], "mode": mode, "is_product_default": False,
            "planned_steps": _STEPS.get(mode, []),
            "note": "dry-run plan only — no training; promotion needs this track's OWN gate, "
                    "never the dense frontier gate"}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--config", default=str(TRACK_CONFIG))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    track = json.loads(Path(args.config).read_text(encoding="utf-8"))
    plan = plan_mode(args.mode, track)
    print(json.dumps({"status": "fail" if plan["errors"] else "dry-run", **plan},
                     ensure_ascii=False, indent=2))
    return 1 if plan["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
