#!/usr/bin/env python3
"""Advisory v7 EmbedFilter gate (pure stdlib). Reads ``outputs/v7-embedfilter/sweep.json`` and
judges whether EmbedFilter is competitive with prefix Matryoshka at equal dims. ADVISORY ONLY —
it never auto-promotes and never fabricates metrics. ``--require-real`` makes a missing/empty
real sweep a hard failure.

Advisory rules (over ACTIVE eval sets only; GerDaLIR/diagnostic excluded):
  * τ=2 / 512-d: mean ΔnDCG@10 and ΔRecall@100 vs full ≥ −0.005 (within tol, or better).
  * τ=4 / 256-d: mean ΔnDCG@10 and ΔRecall@100 vs prefix-256 ≥ 0 (matches or beats prefix).
  * GermanQuAD / DT-test guardrails: no embedfilter (512/256) active regression vs full worse
    than −0.005.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOL = 0.005


def _mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else None


def embedfilter_gate(rows: List[Dict[str, Any]], *, require_real: bool = False) -> Dict[str, Any]:
    """Advisory verdict over the sweep rows. Pure function."""
    active = [r for r in rows if r.get("role") == "active"]
    has_real = bool(active) and all(isinstance(r.get("ndcg@10"), (int, float)) for r in active)
    if require_real and not has_real:
        return {"status": "fail", "advisory": True, "has_real_metrics": has_real, "checks": [],
                "failed": ["real_metrics_present"],
                "recommendation": "No real sweep metrics — run eval_embed_filter_sweep.py."}

    def ef(dim: int) -> List[Dict[str, Any]]:
        return [r for r in active if r.get("method") == "embedfilter" and r.get("dim") == dim]

    checks: List[Dict[str, Any]] = []

    d_ndcg = _mean([r.get("dNDCG10_vs_full") for r in ef(512)])
    d_rec = _mean([r.get("dRecall100_vs_full") for r in ef(512)])
    ok512 = d_ndcg is not None and d_rec is not None and d_ndcg >= -TOL and d_rec >= -TOL
    checks.append({"check": "tau2_512_within_tol_of_full", "status": "pass" if ok512 else "fail",
                   "detail": f"mean Δndcg/full={d_ndcg}, Δrecall/full={d_rec} (need ≥ -{TOL})"})

    p_ndcg = _mean([r.get("dNDCG10_vs_prefix") for r in ef(256)])
    p_rec = _mean([r.get("dRecall100_vs_prefix") for r in ef(256)])
    ok256 = p_ndcg is not None and p_rec is not None and p_ndcg >= -1e-9 and p_rec >= -1e-9
    checks.append({"check": "tau4_256_matches_or_beats_prefix256",
                   "status": "pass" if ok256 else "fail",
                   "detail": f"mean Δndcg/prefix={p_ndcg}, Δrecall/prefix={p_rec} (need ≥ 0)"})

    guard = [r for r in active if r.get("eval_set") in ("germanquad", "dt_test")
             and r.get("method") == "embedfilter" and r.get("dim") in (512, 256)]
    g_vals = [r.get("dNDCG10_vs_full") for r in guard if isinstance(r.get("dNDCG10_vs_full"),
                                                                    (int, float))]
    worst = min(g_vals) if g_vals else 0.0
    okg = worst >= -TOL
    checks.append({"check": "germanquad_dttest_guardrail", "status": "pass" if okg else "fail",
                   "detail": f"worst guardrail Δndcg/full={worst} (need ≥ -{TOL})"})

    failed = [c["check"] for c in checks if c["status"] == "fail"]
    status = "pass" if not failed else "fail"
    rec = ("EmbedFilter is competitive with prefix Matryoshka at equal dim (advisory PASS). "
           "Still no production claim without a saved real sweep + human review."
           if status == "pass" else
           "EmbedFilter does NOT clear the advisory gate — prefer prefix Matryoshka.")
    return {"status": status, "advisory": True, "has_real_metrics": has_real, "checks": checks,
            "failed": failed, "recommendation": rec,
            "note": "advisory only; never auto-promotes; no production claim unless this passes "
                    "on real saved outputs"}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default=str(ROOT / "outputs/v7-embedfilter/sweep.json"))
    ap.add_argument("--require-real", action="store_true",
                    help="fail if the sweep file is missing or has no real metrics")
    ap.add_argument("--output", default=None, help="optional path to write the gate JSON")
    ap.add_argument("--format", choices=["json", "markdown"], default="json")
    args = ap.parse_args(argv)

    sweep_path = pathlib.Path(args.sweep)
    if not sweep_path.exists():
        verdict = {"status": "fail" if args.require_real else "advisory_no_data", "advisory": True,
                   "has_real_metrics": False, "checks": [],
                   "recommendation": f"sweep not found: {args.sweep} — run the sweep first."}
    else:
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        verdict = embedfilter_gate(sweep.get("rows", []), require_real=args.require_real)

    if args.output:
        out = pathlib.Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.format == "markdown":
        print(f"# EmbedFilter advisory gate: **{verdict['status']}**\n")
        for c in verdict.get("checks", []):
            print(f"- {'✅' if c['status'] == 'pass' else '❌'} {c['check']} — {c['detail']}")
        print(f"\n**{verdict['recommendation']}**")
    else:
        print(json.dumps({"status": verdict["status"], "failed": verdict.get("failed", []),
                          "has_real_metrics": verdict.get("has_real_metrics")}, ensure_ascii=False))
    # exit nonzero only when require-real and not passing (so it's a usable CLI gate)
    return 0 if verdict["status"] == "pass" else (1 if args.require_real else 0)


if __name__ == "__main__":
    raise SystemExit(main())
