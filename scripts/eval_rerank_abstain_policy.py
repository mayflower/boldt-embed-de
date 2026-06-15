#!/usr/bin/env python3
"""Evaluate a fitted rerank-or-abstain policy on eval sets and gate it. Reports per set:
first_stage / always_rerank / policy nDCG@10, deltas, abstain/rerank rates, catastrophic_drop_rate,
and metrics by hardness_bucket (oracle/labels used for ANALYSIS ONLY, never at inference).

Single set:  --eval-lists path.jsonl --output eval.json
Multi set:   --eval-lists webfaq=.. germanquad=.. dt_test=.. --out-dir DIR   (writes eval_*.json + gate.{json,md})
`--dry-run` imports no torch.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import rerank_abstain as RA  # noqa: E402


def _read(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True, help="fit_report.json (best_params)")
    ap.add_argument("--eval-lists", nargs="+", required=True, help="path OR name=path (repeatable)")
    ap.add_argument("--output", default=None, help="single-set output json")
    ap.add_argument("--out-dir", default=None, help="multi-set dir for eval_*.json + gate.{json,md}")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fit = json.loads(pathlib.Path(args.policy).read_text(encoding="utf-8"))
    policy = fit.get("policy", "combined_policy")
    params = fit.get("best_params", {})

    specs = []
    for s in args.eval_lists:
        if "=" in s:
            name, path = s.split("=", 1)
        else:
            name, path = pathlib.Path(s).stem.replace("rag_", "").replace("_lists_scored", ""), s
        specs.append((name, path))

    reports, always = {}, {}
    for name, path in specs:
        rows = _read(path)
        rep = RA.evaluate_policy(rows, policy, params)
        al = RA.evaluate_policy(rows, "always_rerank", {})
        rep["always_rerank_delta_vs_first_stage"] = al["delta_vs_first_stage"]
        rep["always_rerank_catastrophic_drop_rate"] = al["catastrophic_drop_rate"]
        rep["eval_set"] = name
        reports[name] = rep
        always[name] = al
        print(f"[eval-abstain] {name}: policyΔ {rep['delta_vs_first_stage']:+} vs first-stage "
              f"(always_rerankΔ {al['delta_vs_first_stage']:+}); abstain {rep['abstain_rate']} "
              f"catastrophic {rep['catastrophic_drop_rate']}", file=sys.stderr)

    if args.dry_run:
        assert "torch" not in sys.modules, "dry-run must not import torch"

    if args.out_dir:
        out = pathlib.Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, rep in reports.items():
            (out / f"eval_{name}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
        gate = RA.policy_gate(reports, always)
        (out / "gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        md = [f"# rerank-or-abstain policy gate: **{gate['status']}**", "",
              f"Policy `{policy}` params `{json.dumps(params)}`", "",
              "| eval set | policy Δ vs first-stage | always_rerank Δ | medium+hard Δ | "
              "abstain | catastrophic | ",
              "|---|--:|--:|--:|--:|--:|"]
        for name, rep in reports.items():
            md.append(f"| {name} | {rep['delta_vs_first_stage']:+} | "
                      f"{rep['always_rerank_delta_vs_first_stage']:+} | {rep.get('medium_hard_delta')} | "
                      f"{rep['abstain_rate']} | {rep['catastrophic_drop_rate']} |")
        md += ["", "## Checks", ""]
        for c in gate["checks"]:
            md.append(f"- {'✅' if c['status'] == 'pass' else '❌'} {c['check']}: {c['detail']}")
        (out / "gate.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"[eval-abstain] gate={gate['status']} -> {out}/gate.json")
        if args.dry_run:
            print("dry-run-ok (no ML imports)")
        return 0 if gate["status"] == "pass" else 1

    # single set
    name = specs[0][0]
    outp = pathlib.Path(args.output or f"outputs/v5-small-rag/abstain/eval_{name}.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(reports[name], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval-abstain] {name} -> {outp}")
    if args.dry_run:
        print("dry-run-ok (no ML imports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
