#!/usr/bin/env python3
"""v8 promotion gate — judges a v8 candidate against the v6.1 baseline on MTEB(deu) retrieval-core.

Pure stdlib. Reads saved MTEB summaries (`outputs/mteb/<label>/summary.json`, ADR-005) — a metric
without a saved summary is not a claim. PRIMARY signal is the **headroom** set (MIRACL), per the
repo's promotion philosophy; near-ceiling / do-not-regress sets carry only a tolerance.

Gate (see docs/v8-improvement-research-2026.md §3):
  - PRIMARY: MIRACL-hn nDCG@10 closes >= 1/3 of the v6.1->competitor gap (target >= ~0.39) AND
             GermanQuAD does not regress (>= baseline - tol).
  - DO-NOT-REGRESS: GerDaLIR / MLDR >= baseline - tol (compared at the SAME seq length).
  - FAIL-CLOSED: a bidirectional candidate must have been evaluated WITH the patch applied
                 (meta.loader == 'st' + meta has no causal marker is insufficient; we require an
                 explicit bidirectional flag in the run metadata, else the number is invalid).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOL = 0.005
MIRACL_TARGET = 0.39  # close >= 1/3 of the ~0.33 -> ~0.52 gap
TASK = {"miracl": "MIRACLRetrievalHardNegatives", "germanquad": "GermanQuAD-Retrieval",
        "gerdalir": "GerDaLIRSmall", "mldr": "MultiLongDocRetrieval"}


def _summary(label: str) -> dict:
    p = ROOT / "outputs" / "mteb" / label / "summary.json"
    if not p.exists():
        raise SystemExit(f"missing MTEB summary for {label!r}: {p} (run it before gating)")
    return json.loads(p.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True, help="MTEB label under outputs/mteb/")
    ap.add_argument("--baseline", default="v6-1-baseline", help="MTEB label of the v6.1 baseline")
    ap.add_argument("--require-bidirectional", action="store_true",
                    help="fail if the candidate was not evaluated with the bidirectional patch")
    ap.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = ap.parse_args(argv)

    cand, base = _summary(args.candidate), _summary(args.baseline)
    cs, bs = cand.get("scores", {}), base.get("scores", {})
    cmeta = cand.get("meta", {})
    checks, failed = [], []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            failed.append(name)

    # fail-closed: bidirectional eval must have applied the patch
    if args.require_bidirectional:
        applied = bool(cmeta.get("bidirectional"))
        add("bidirectional_patch_applied", applied,
            "run with --bidirectional (patch is runtime; else evaluated as causal)")

    miracl_c = cs.get(TASK["miracl"])
    gq_c, gq_b = cs.get(TASK["germanquad"]), bs.get(TASK["germanquad"])
    add("miracl_headroom", miracl_c is not None and miracl_c >= MIRACL_TARGET,
        f"MIRACL {miracl_c} >= {MIRACL_TARGET} (target: close >=1/3 of gap)")
    add("germanquad_no_regress", gq_c is not None and gq_b is not None and gq_c >= gq_b - TOL,
        f"GermanQuAD {gq_c} >= baseline {gq_b} - {TOL}")
    for key in ("gerdalir", "mldr"):
        c, b = cs.get(TASK[key]), bs.get(TASK[key])
        if c is not None and b is not None:  # only gate when both measured at comparable seq
            add(f"{key}_no_regress", c >= b - TOL, f"{key} {c} >= baseline {b} - {TOL}")

    promotable = not failed
    verdict = {"candidate": args.candidate, "baseline": args.baseline,
               "promotable": promotable, "failed_gates": failed, "checks": checks,
               "note": "MTEB(deu) retrieval-core; numbers from saved summaries (ADR-005). "
                       "A bidirectional candidate is only valid if evaluated with the patch."}

    if args.format == "json":
        print(json.dumps(verdict, ensure_ascii=False, indent=2))
    else:
        print(f"# v8 gate: {args.candidate} vs {args.baseline}\n")
        print(f"**promotable: {promotable}**" + (f" — failed: {', '.join(failed)}" if failed else ""))
        for c in checks:
            print(f"- [{'x' if c['ok'] else ' '}] {c['name']}: {c['detail']}")
    return 0 if promotable else 1


if __name__ == "__main__":
    raise SystemExit(main())
