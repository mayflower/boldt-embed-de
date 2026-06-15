#!/usr/bin/env python3
"""Evaluate a fitted bounded reranking policy on eval set(s). Reports policy_name, thresholds,
abstain/lock rate, avg max displacement, nDCG@10 before/after, delta, catastrophic_drop_rate,
by-hardness-bucket, and per-query top catastrophic examples. Labels/oracle/buckets are used for
ANALYSIS ONLY — never by the policy decision. `--dry-run` imports no torch.

Single set: --eval-lists path.jsonl --output eval.json
Multi set:  --eval-lists webfaq=.. germanquad=.. dt_test=.. --out-dir DIR
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import bounded_rerank as BR  # noqa: E402


def _read(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True, help="fit_report.json (policy + best_params)")
    ap.add_argument("--eval-lists", nargs="+", required=True, help="path OR name=path (repeatable)")
    ap.add_argument("--output", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fit = json.loads(pathlib.Path(args.policy).read_text(encoding="utf-8"))
    policy = fit.get("policy", "identity")
    params = fit.get("best_params", {})

    specs = []
    for s in args.eval_lists:
        name, path = (s.split("=", 1) if "=" in s
                      else (pathlib.Path(s).stem.replace("rag_", "").replace("_lists_scored", ""), s))
        specs.append((name, path))

    reports = {}
    for name, path in specs:
        rows = _read(path)
        rep = BR.evaluate_policy(rows, policy, params)
        rep["policy_name"] = policy
        rep["selected_thresholds"] = params
        al = BR.evaluate_policy(rows, "always_rerank", {}, with_buckets=False)
        rep["always_rerank_delta_vs_first_stage"] = al["delta_vs_first_stage"]
        rep["always_rerank_catastrophic_drop_rate"] = al["catastrophic_drop_rate"]
        rep["eval_set"] = name
        reports[name] = rep
        print(f"[eval-bounded] {name}: {policy} Δ {rep['delta_vs_first_stage']:+} "
              f"(always_rerank Δ {al['delta_vs_first_stage']:+}); catastrophic "
              f"{rep['catastrophic_drop_rate']} (always {al['catastrophic_drop_rate']}); "
              f"lock {rep['lock_rate']} avg_disp {rep['avg_max_displacement']}", file=sys.stderr)

    if args.dry_run:
        assert "torch" not in sys.modules, "dry-run must not import torch"

    if args.out_dir:
        out = pathlib.Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, rep in reports.items():
            (out / f"eval_{name}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
        print(f"[eval-bounded] wrote {len(reports)} reports -> {out}")
    else:
        name = specs[0][0]
        outp = pathlib.Path(args.output or f"outputs/v5-small-rag/bounded/eval_{name}.json")
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(reports[name], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval-bounded] {name} -> {outp}")
    if args.dry_run:
        print("dry-run-ok (no ML imports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
