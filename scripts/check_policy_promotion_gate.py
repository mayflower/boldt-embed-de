#!/usr/bin/env python3
"""Promotion gate for the FROZEN bounded rerank policy (stdlib, no ML). Reads the per-set eval
reports from eval_policy_gate_v5 and decides promotion. The decision is about the POLICY, not the
raw model: raw always-rerank can never pass, and a policy tuned on a guardrail (GerDaLIR ignored;
GermanQuAD/DT-test tuned-markers) fails closed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

GATE = {
    "webfaq_min_policy_delta": 0.05,
    "near_ceiling_max_catastrophic": 0.03, "near_ceiling_min_policy_delta": -0.005,
    "germanquad_max_catastrophic": 0.03, "germanquad_min_policy_delta": -0.005,
    "dt_test_max_catastrophic": 0.02, "dt_test_min_policy_delta": -0.005,
}
DIAGNOSTIC = {"gerdalir", "legal"}


def promotion_gate(reports: dict, *, tuned_on_guardrail: bool = False) -> dict:
    """reports: set name -> eval_policy_gate_v5 report. Returns the gate decision."""
    checks = []

    def chk(name, ok, detail):
        checks.append({"check": name, "status": "pass" if ok else "fail", "detail": detail})

    # raw always-rerank can NEVER be promoted as a policy.
    for name, r in reports.items():
        if name in DIAGNOSTIC:
            continue
        if r.get("ranking_mode") == "raw_rerank":
            chk("not_raw_rerank", False, f"{name} evaluated in raw_rerank mode — cannot be promoted")
    # a policy tuned on a public guardrail fails closed
    chk("not_tuned_on_guardrail", not tuned_on_guardrail,
        "policy must not be tuned on GermanQuAD/DT-test")

    wf = reports.get("webfaq")
    if wf:
        chk("webfaq_policy_delta", wf["policy_delta"] >= GATE["webfaq_min_policy_delta"] - 1e-9,
            f"{wf['policy_delta']:+.4f} (min +{GATE['webfaq_min_policy_delta']})")
    for s, role in (("near_ceiling", "near-ceiling guardrail"), ("germanquad", "guardrail"),
                    ("dt_test", "guardrail")):
        r = reports.get(s)
        if not r:
            continue
        chk(f"{s}_catastrophic", r["catastrophic_drop_rate"] <= GATE[f"{s}_max_catastrophic"] + 1e-9,
            f"{r['catastrophic_drop_rate']:.4f} (max {GATE[f'{s}_max_catastrophic']})")
        chk(f"{s}_policy_delta", r["policy_delta"] >= GATE[f"{s}_min_policy_delta"] - 1e-9,
            f"{r['policy_delta']:+.4f} (min {GATE[f'{s}_min_policy_delta']})")
    # policy must beat raw_rerank on GermanQuAD and near_ceiling
    for s in ("germanquad", "near_ceiling"):
        r = reports.get(s)
        if r:
            chk(f"{s}_beats_raw_rerank", r["policy_delta"] > r["raw_delta"] - 1e-9,
                f"policy {r['policy_delta']:+.4f} vs raw {r['raw_delta']:+.4f}")

    failing = [c for c in checks if c["status"] == "fail"]
    return {"status": "pass" if not failing else "fail", "checks": checks, "failing": failing,
            "thresholds": GATE, "ignored_diagnostic_sets": sorted(set(reports) & DIAGNOSTIC),
            "raw_always_rerank_recommended": False}


def _load_reports(eval_dir):
    d = pathlib.Path(eval_dir)
    reports = {}
    for p in sorted(d.glob("eval_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        reports[r.get("eval_set", p.stem.replace("eval_", ""))] = r
    return reports


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--markdown", required=True)
    args = ap.parse_args()

    reports = _load_reports(args.eval_dir)
    if not reports:
        print(f"ERROR: no eval_*.json in {args.eval_dir}", file=sys.stderr)
        return 2
    tuned = (pathlib.Path(args.eval_dir) / "TUNED_ON_GUARDRAIL").exists() or \
        any(r.get("tuned_on") in ("germanquad", "dt_test") for r in reports.values())
    gate = promotion_gate(reports, tuned_on_guardrail=tuned)

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(gate, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    md = [f"# Frozen bounded-policy promotion gate: **{gate['status']}**", "",
          f"_Decision is about the policy `bounded_margin_override_v1`, never raw always-rerank. "
          f"Diagnostic sets ignored: {gate['ignored_diagnostic_sets'] or 'none'}._", "",
          "| eval set | role | policy Δ | raw Δ | catastrophic | mode |",
          "|---|---|--:|--:|--:|---|"]
    for name, r in sorted(reports.items()):
        md.append(f"| {name} | {r.get('role')} | {r.get('policy_delta'):+} | {r.get('raw_delta'):+} | "
                  f"{r.get('catastrophic_drop_rate')} | {r.get('ranking_mode')} |")
    md += ["", "## Checks", ""]
    for c in gate["checks"]:
        md.append(f"- {'✅' if c['status'] == 'pass' else '❌'} {c['check']}: {c['detail']}")
    if gate["status"] != "pass":
        md += ["", f"**Verdict: NOT promoted** — failing: {[c['check'] for c in gate['failing']]}. "
               "Reranker stays Experimental."]
    else:
        md += ["", "**Verdict: PROMOTABLE with the bounded policy only** (raw always-rerank remains "
               "not recommended)."]
    pathlib.Path(args.markdown).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[policy-gate] status={gate['status']} failing={[c['check'] for c in gate['failing']]} "
          f"-> {args.output}")
    return 0 if gate["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
