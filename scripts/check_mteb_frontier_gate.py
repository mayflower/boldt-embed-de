#!/usr/bin/env python3
"""v8+ multi-task promotion gate — does a candidate BEAT the same-size-peer frontier on MTEB(deu)?

Success bar (per the v8+ plan): beat the same-class peers (e5-base 278M, LFM2.5 350M) on the
AGGREGATE of the four retrieval-core tasks, win individual tasks where reachable, and NEVER regress
vs the model's own baseline. Qwen3-0.6B (596M) is a stretch reference, not a gate.

Pure stdlib. Reads only saved `outputs/mteb/<label>/summary.json` (ADR-005) — a metric without a
saved summary is not a claim. PROTECTED (`check_*`): run OUTSIDE the AutoResearch loop, after eval.

PASS iff:
  - candidate aggregate (mean nDCG@10 over the 4 tasks) >= the peer-frontier aggregate, AND
  - no task regresses below baseline - tol (do-not-regress guardrail), AND
  - WebFAQ recall@100 (if a summary is supplied) >= the WebFAQ floor (primary do-not-regress), AND
  - candidate eval is leakage-clean if its run metadata records leakage (fail-closed).
The per-task "beats peer" flags + count are reported (informational; matching all is the stretch).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["GermanQuAD-Retrieval", "GerDaLIRSmall", "MIRACLRetrievalHardNegatives",
         "MultiLongDocRetrieval"]
SHORT = {"GermanQuAD-Retrieval": "GermanQuAD", "GerDaLIRSmall": "GerDaLIR",
         "MIRACLRetrievalHardNegatives": "MIRACL", "MultiLongDocRetrieval": "MLDR"}


def _scores(label: str, required: bool = True) -> dict:
    p = ROOT / "outputs" / "mteb" / label / "summary.json"
    if not p.exists():
        if required:
            raise SystemExit(f"missing MTEB summary for {label!r}: {p} (run /ar-mteb first)")
        return {}
    return json.loads(p.read_text(encoding="utf-8")).get("scores", {})


def _meta(label: str) -> dict:
    p = ROOT / "outputs" / "mteb" / label / "summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8")).get("meta", {})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True, help="MTEB label under outputs/mteb/")
    ap.add_argument("--peers", default="e5-base,lfm2.5",
                    help="same-size-peer MTEB labels to beat (comma-separated)")
    ap.add_argument("--baseline", default="v6-1-baseline-512",
                    help="own-baseline label for the do-not-regress guardrail")
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = ap.parse_args(argv)

    cand = _scores(args.candidate)
    peers = {p: _scores(p) for p in (x.strip() for x in args.peers.split(",")) if p}
    base = _scores(args.baseline, required=False)
    cmeta = _meta(args.candidate)

    def agg(s):
        vals = [s[t] for t in TASKS if isinstance(s.get(t), (int, float))]
        return sum(vals) / len(vals) if vals else None

    peer_front = {t: max((p[t] for p in peers.values() if isinstance(p.get(t), (int, float))),
                         default=None) for t in TASKS}
    cand_agg = agg(cand)
    peer_agg = agg(peer_front)

    checks, failed, per_task = [], [], []
    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            failed.append(name)

    add("beats_peer_frontier_aggregate",
        cand_agg is not None and peer_agg is not None and cand_agg >= peer_agg,
        f"candidate mean nDCG@10 {cand_agg} >= peer-frontier mean {peer_agg}")

    # Fail-closed: a candidate can never be promotable without a real own-baseline to grade the
    # do-not-regress guardrail against. A missing baseline summary disables those per-task checks
    # below (b is None), so without this the guardrail would silently vanish.
    base_has_scores = any(isinstance(base.get(t), (int, float)) for t in TASKS)
    add("baseline_present", base_has_scores,
        f"own-baseline {args.baseline!r} summary present with task scores "
        f"(do-not-regress guardrail requires it; run /ar-mteb to create it)")

    beats = 0
    for t in TASKS:
        c, pf, b = cand.get(t), peer_front.get(t), base.get(t)
        won = isinstance(c, (int, float)) and isinstance(pf, (int, float)) and c >= pf
        beats += 1 if won else 0
        per_task.append({"task": SHORT[t], "candidate": c, "peer_best": pf,
                         "baseline": b, "beats_peer": won})
        if isinstance(c, (int, float)) and isinstance(b, (int, float)):  # do-not-regress vs own base
            add(f"no_regress_{SHORT[t]}", c >= b - args.tol,
                f"{SHORT[t]} {c} >= baseline {b} - {args.tol}")

    # WebFAQ primary do-not-regress (optional, if a WebFAQ summary is supplied as candidate meta)
    wf = cmeta.get("webfaq_recall100")
    wf_floor = cmeta.get("webfaq_floor", 0.97)
    if wf is not None:
        add("webfaq_no_regress", wf >= wf_floor, f"WebFAQ recall@100 {wf} >= {wf_floor}")

    # leakage fail-closed (only if the candidate's training/eval metadata records a status)
    lk = cmeta.get("leakage_status") or (cmeta.get("leakage") or {}).get("status")
    if lk is not None:
        add("leakage_clean", lk in ("clean", "scanned_clean"), f"leakage_status={lk}")

    promotable = not failed
    verdict = {"candidate": args.candidate, "peers": list(peers), "baseline": args.baseline,
               "promotable": promotable, "failed_gates": failed,
               "candidate_aggregate": cand_agg, "peer_frontier_aggregate": peer_agg,
               "tasks_beating_peers": f"{beats}/{len(TASKS)}", "per_task": per_task,
               "checks": checks,
               "note": "MTEB(deu) retrieval-core; saved summaries only (ADR-005). PASS = beats the "
                       "same-size-peer aggregate AND no task regresses vs own baseline."}

    if args.format == "json":
        print(json.dumps(verdict, ensure_ascii=False, indent=2))
    else:
        print(f"# v8+ frontier gate: {args.candidate} vs peers {list(peers)}\n")
        print(f"**promotable: {promotable}**" + (f" — failed: {', '.join(failed)}" if failed else ""))
        print(f"aggregate: candidate {cand_agg} vs peer-frontier {peer_agg} | "
              f"beats peers on {beats}/{len(TASKS)} tasks\n")
        print("| task | candidate | peer-best | baseline | beats peer |")
        print("|---|---|---|---|---|")
        for r in per_task:
            print(f"| {r['task']} | {r['candidate']} | {r['peer_best']} | {r['baseline']} | "
                  f"{'✅' if r['beats_peer'] else '—'} |")
        for c in checks:
            print(f"- [{'x' if c['ok'] else ' '}] {c['name']}: {c['detail']}")
    return 0 if promotable else 1


if __name__ == "__main__":
    raise SystemExit(main())
