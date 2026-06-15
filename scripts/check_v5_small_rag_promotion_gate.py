#!/usr/bin/env python3
"""v5 small-RAG promotion gate (stdlib, no ML). Reads the per-set abstention eval reports written
by `eval_rerank_abstain_policy.py` (eval_webfaq.json / eval_germanquad.json / eval_dt_test.json)
and runs the policy gate (`rerank_abstain.policy_gate`). Pass => the reranker may be recommended
ONLY with the abstention policy; fail => keep experimental. GermanQuAD/DT-test are guardrails.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import rerank_abstain as RA  # noqa: E402


def _load(p):
    p = pathlib.Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--abstain-dir", default=None)
    ap.add_argument("--bounded-dir", default=None,
                    help="bounded-policy eval dir -> task's 7-check bounded gate")
    ap.add_argument("--output", required=True)
    ap.add_argument("--markdown", required=True)
    args = ap.parse_args()

    if not (args.abstain_dir or args.bounded_dir):
        print("ERROR: pass --abstain-dir or --bounded-dir", file=sys.stderr)
        return 2
    d = pathlib.Path(args.bounded_dir or args.abstain_dir)
    reports, always = {}, {}
    missing = []
    for name in ("webfaq", "germanquad", "dt_test"):
        rep = _load(d / f"eval_{name}.json")
        if rep is None:
            missing.append(f"eval_{name}.json")
            continue
        reports[name] = rep
        # reconstruct the always_rerank reference recorded by the eval script
        always[name] = {
            "delta_vs_first_stage": rep.get("always_rerank_delta_vs_first_stage"),
            "catastrophic_drop_rate": rep.get("always_rerank_catastrophic_drop_rate"),
        }
    if missing:
        print(f"ERROR: missing eval reports: {missing}", file=sys.stderr)
        return 2

    if args.bounded_dir:
        from boldt_embed import bounded_rerank as BR
        gate = BR.bounded_policy_gate(reports)
    else:
        gate = RA.policy_gate(reports, always)
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(gate, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    md = [f"# v5 small-RAG abstention promotion gate: **{gate['status']}**", "",
          "| eval set | policy Δ vs first-stage | always_rerank Δ | medium+hard Δ | "
          "abstain | catastrophic |", "|---|--:|--:|--:|--:|--:|"]
    for name in ("webfaq", "germanquad", "dt_test"):
        r = reports[name]
        md.append(f"| {name} | {r['delta_vs_first_stage']:+} | "
                  f"{r.get('always_rerank_delta_vs_first_stage')} | {r.get('medium_hard_delta')} | "
                  f"{r['abstain_rate']} | {r['catastrophic_drop_rate']} |")
    md += ["", "## Checks", ""]
    for c in gate["checks"]:
        md.append(f"- {'✅' if c['status'] == 'pass' else '❌'} {c['check']}: {c['detail']}")
    if gate["status"] != "pass":
        md += ["", f"**Verdict: NOT promoted** — failing: "
               f"{', '.join(c['check'] for c in gate['failing'])}. Reranker stays Experimental."]
    pathlib.Path(args.markdown).write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[v5-gate] status={gate['status']} "
          f"failing={[c['check'] for c in gate['failing']]} -> {args.output}")
    return 0 if gate["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
