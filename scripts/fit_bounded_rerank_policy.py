#!/usr/bin/env python3
"""Fit a bounded reranking policy on a DEV split ONLY (dev labels allowed). Grid search prefers
candidates whose dev catastrophic-drop rate is <= target, maximizing dev nDCG@10. GermanQuAD/DT-test
are NEVER passed here. `--dry-run` imports no torch (pure stdlib). Writes fit_report.{json,md}.
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
    ap.add_argument("--dev-lists", required=True, help="DEV scored lists; NOT a guardrail")
    ap.add_argument("--output", required=True)
    ap.add_argument("--markdown", required=True)
    ap.add_argument("--catastrophic-target", type=float, default=0.03)
    ap.add_argument("--grid-search", action="store_true", help="(default) grid-search on dev")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dev = _read(args.dev_lists)
    fit = BR.grid_search(dev, catastrophic_target=args.catastrophic_target)
    if args.dry_run:
        assert "torch" not in sys.modules, "dry-run must not import torch"

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(fit, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    d = fit["dev_metrics"]
    md = [f"# bounded rerank policy fit — **{fit['policy']}**", "",
          f"Fit on DEV ONLY: `{args.dev_lists}` ({len(dev)} lists). Guardrails not used.", "",
          f"- selected params: `{json.dumps(fit['best_params'])}`",
          f"- observable safety (high-conf dev top-1 keep): {fit['safety']}",
          f"- selected a safe policy: {fit['safety']['selected_safe']}",
          f"- dev nDCG@10: {d['dev_ndcg']} (delta {d['dev_delta']:+}), "
          f"dev catastrophic: {d['catastrophic']}, hc_top1_keep: {d.get('hc_top1_keep')}", "",
          "## Top dev candidates (highest high-conf top-1 keep, then highest nDCG)", "",
          "| policy | params | dev nDCG@10 | dev delta | hc_top1_keep | catastrophic |",
          "|---|---|--:|--:|--:|--:|"]
    for t in fit["trials"][:15]:
        md.append(f"| {t['policy']} | {json.dumps(t['params'])} | {t['dev_ndcg']} | "
                  f"{t['dev_delta']:+} | {t.get('hc_top1_keep')} | {t['catastrophic']} |")
    pathlib.Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.markdown).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[fit-bounded] policy={fit['policy']} params={fit['best_params']} "
          f"dev_delta={d['dev_delta']:+} dev_catastrophic={d['catastrophic']} "
          f"safe={fit['safety']['selected_safe']} -> {args.output}")
    if args.dry_run:
        print("dry-run-ok (no ML imports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
