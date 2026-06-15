#!/usr/bin/env python3
"""Fit the rerank-or-abstain policy thresholds on a DEV split ONLY (WebFAQ dev or a private/local
dev). Grid is data-adaptive (quantiles of the dev gap features). GermanQuAD/DT-test are NEVER
passed here — guardrails cannot influence threshold selection. Writes fit_report.{json,md}.
`--dry-run` imports no torch (the whole module is stdlib).
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


def _quantiles(values, qs):
    xs = sorted(values)
    if not xs:
        return [0.0]
    out = []
    for q in qs:
        i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
        out.append(round(xs[i], 6))
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev-lists", required=True, help="DEV candidate lists (scored); NOT a guardrail")
    ap.add_argument("--policy", default="combined_policy", choices=list(RA.POLICIES))
    ap.add_argument("--output", required=True)
    ap.add_argument("--markdown", required=True)
    ap.add_argument("--alphas", default="1.0,0.7,0.5")
    ap.add_argument("--max-displacements", default="3,5,10")
    ap.add_argument("--abstain-target", type=float, default=None)
    ap.add_argument("--grid-search", action="store_true",
                    help="(default behavior) grid-search thresholds on the dev split")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dev = _read(args.dev_lists)
    feats = [RA.extract_features(r) for r in dev]
    qs = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]
    fs_gaps = _quantiles([f["first_stage_top1_top2_gap"] for f in feats], qs) or [0.0]
    rr_gaps = _quantiles([f["reranker_top1_top2_gap"] for f in feats], qs) or [0.0]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    maxd = [int(x) for x in args.max_displacements.split(",") if x.strip()]

    fit = RA.grid_search(dev, policy=args.policy, fs_gaps=fs_gaps, rr_gaps=rr_gaps,
                         alphas=alphas, max_displacements=maxd, abstain_target=args.abstain_target)
    # reference baselines on dev
    fit["dev_always_rerank"] = RA.evaluate_policy(dev, "always_rerank", {})["delta_vs_first_stage"]
    fit["dev_never_rerank"] = RA.evaluate_policy(dev, "never_rerank", {})["delta_vs_first_stage"]
    fit["grid"] = {"fs_gaps": fs_gaps, "rr_gaps": rr_gaps, "alphas": alphas, "max_displacements": maxd}

    if args.dry_run:
        assert "torch" not in sys.modules, "dry-run must not import torch"

    outp = pathlib.Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(fit, ensure_ascii=False, indent=2), encoding="utf-8")

    d = fit["dev_metrics"]
    md = [f"# rerank-or-abstain fit ({args.policy})", "",
          f"Fit on DEV ONLY: `{args.dev_lists}` ({len(dev)} lists). Guardrails not used.", "",
          "## Best params", "",
          f"```json\n{json.dumps(fit['best_params'], indent=2)}\n```", "",
          "## Dev metrics (best policy)", "",
          f"- dev nDCG@10: {d['policy_ndcg@10']} (first-stage {d['first_stage_ndcg@10']}, "
          f"always_rerank {d['always_rerank_ndcg@10']})",
          f"- delta vs first-stage: {d['delta_vs_first_stage']:+}",
          f"- delta vs always_rerank: {d['delta_vs_always_rerank']:+}",
          f"- abstain_rate: {d['abstain_rate']}  rerank_rate: {d['rerank_rate']}",
          f"- catastrophic_drop_rate: {d['catastrophic_drop_rate']}", "",
          f"_grid: fs_gaps={fs_gaps}, rr_gaps={rr_gaps}, alphas={alphas}, max_disp={maxd}; "
          f"{fit['n_trials']} trials_"]
    pathlib.Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.markdown).write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[fit-abstain] policy={args.policy} best={fit['best_params']} "
          f"dev_delta={d['delta_vs_first_stage']:+} abstain={d['abstain_rate']} -> {outp}")
    if args.dry_run:
        print("dry-run-ok (no ML imports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
