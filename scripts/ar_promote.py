#!/usr/bin/env python3
"""MTEB(deu) promotion step — run the (protected) frontier gate and write an auditable verdict.

This is the AutoResearch ``promotion`` trial. It does NOT re-implement or weaken the gate: it invokes
``scripts/check_mteb_frontier_gate.py`` and records its verdict as ``promotion_verdict.json`` +
``promotion_report.md``. Fail-closed: a missing candidate/peer/baseline summary makes the gate fail,
and so does this step. No number is claimed beyond the saved ``outputs/mteb/<label>/summary.json``.

    python scripts/ar_promote.py --candidate v8-diverse-causal --format markdown
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "check_mteb_frontier_gate.py"


def run_gate(candidate: str, peers: str, baseline: str, tol: float,
             mteb_root: Optional[str] = None) -> Dict[str, Any]:
    """Invoke the protected frontier gate as a subprocess; return its JSON verdict (or a fail dict)."""
    cmd = [sys.executable, str(GATE), "--candidate", candidate, "--peers", peers,
           "--baseline", baseline, "--tol", str(tol), "--format", "json"]
    if mteb_root:
        cmd += ["--mteb-root", mteb_root]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    out = (proc.stdout or "").strip()
    try:
        verdict = json.loads(out)
    except json.JSONDecodeError:
        # the gate fail-closes on a missing summary by raising SystemExit (non-JSON stderr)
        err = (proc.stderr or out or "gate produced no JSON").strip()
        # classify: a missing PEER/BASELINE summary is a setup/config problem, NOT the candidate
        # failing the frontier — don't misattribute it to the candidate.
        peers_named = [p for p in (x.strip() for x in peers.split(",")) if p and p in err]
        if f"for {candidate!r}" in err:
            kind = "candidate_summary_missing"
        elif baseline in err or peers_named:
            kind = "setup_error_missing_peer_or_baseline_summary"
        else:
            kind = "gate_error"
        return {"candidate": candidate, "promotable": False,
                "failed_gates": [kind], "error_kind": kind, "gate_returncode": proc.returncode,
                "error": err}
    verdict["gate_returncode"] = proc.returncode
    return verdict


def render_report(verdict: Dict[str, Any]) -> str:
    lines = ["# MTEB(deu) promotion report", ""]
    lines.append(f"Candidate: `{verdict.get('candidate')}`  ·  promotable: "
                 f"**{verdict.get('promotable')}**")
    if verdict.get("error"):
        kind = verdict.get("error_kind", "gate_error")
        if kind == "setup_error_missing_peer_or_baseline_summary":
            lines += ["", "SETUP ERROR (not a candidate failure): a peer/baseline MTEB summary is "
                          f"missing — run those evals first. Detail: {verdict['error']}"]
        elif kind == "candidate_summary_missing":
            lines += ["", f"FAIL: the candidate has no MTEB summary yet — run /ar-mteb-trial first. "
                          f"Detail: {verdict['error']}"]
        else:
            lines += ["", f"FAIL (gate error): {verdict['error']}"]
        return "\n".join(lines) + "\n"
    lines.append(f"Candidate aggregate: {verdict.get('candidate_aggregate')}  ·  "
                 f"peer-frontier aggregate: {verdict.get('peer_frontier_aggregate')}  ·  "
                 f"tasks beating peers: {verdict.get('tasks_beating_peers')}")
    lines += ["", "| task | candidate | peer-best | baseline | beats peer |",
              "|---|---|---|---|---|"]
    for r in verdict.get("per_task", []):
        lines.append(f"| {r['task']} | {r['candidate']} | {r['peer_best']} | "
                     f"{r['baseline']} | {r['beats_peer']} |")
    lines += ["", f"Failed gates: {verdict.get('failed_gates') or 'none'}", ""]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--peers", default="e5-base,lfm2.5")
    ap.add_argument("--baseline", default="v6-1-baseline")
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--mteb-root", default=None)
    ap.add_argument("--out", default="outputs/autoresearch/mteb")
    ap.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = ap.parse_args(argv)

    verdict = run_gate(args.candidate, args.peers, args.baseline, args.tol, mteb_root=args.mteb_root)
    report = render_report(verdict)

    out_dir = (Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out) / args.candidate
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "promotion_verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "promotion_report.md").write_text(report, encoding="utf-8")

    if args.format == "json":
        print(json.dumps(verdict, ensure_ascii=False, indent=2))
    else:
        print(report)
    return 0 if verdict.get("promotable") else 1


if __name__ == "__main__":
    raise SystemExit(main())
