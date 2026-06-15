#!/usr/bin/env python3
"""Analyze the difficulty (hardness) distribution of a fixed candidate-list set (stdlib, no ML).

For each list computes first_stage/oracle nDCG@10, positive_in_top_10/50, num_candidates,
num_candidate_sources, and a hardness bucket (no_room/easy/medium/hard/impossible), then reports
the bucket distribution so you can see how much real headroom a set has BEFORE judging a reranker.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import hardness_aware_eval as H  # noqa: E402


def _read(path: pathlib.Path) -> list:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").split("\n") if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-lists", required=True)
    ap.add_argument("--eval-set", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = pathlib.Path(args.candidate_lists)
    if not path.exists():
        print(f"ERROR: candidate lists not found: {path}", file=sys.stderr)
        return 2
    name = args.eval_set or path.stem
    metrics = [m for m in (H.list_metrics(r) for r in _read(path)) if m]
    if not metrics:
        print("ERROR: no candidate lists with candidates", file=sys.stderr)
        return 1

    n = len(metrics)
    buckets: dict = {}
    for m in metrics:
        buckets[m["hardness_bucket"]] = buckets.get(m["hardness_bucket"], 0) + 1

    def _avg(key):
        return round(sum(m[key] for m in metrics) / n, 4)

    report = {
        "eval_set": name, "n_queries": n,
        "bucket_counts": dict(sorted(buckets.items())),
        "bucket_fractions": {b: round(c / n, 4) for b, c in sorted(buckets.items())},
        "mean_first_stage_ndcg@10": _avg("first_stage_ndcg@10"),
        "mean_oracle_ndcg@10": _avg("oracle_ndcg@10"),
        "positive_in_top_10_rate": _avg("positive_in_top_10"),
        "positive_in_top_50_rate": _avg("positive_in_top_50"),
        "mean_num_candidates": _avg("num_candidates"),
        "mean_num_candidate_sources": _avg("num_candidate_sources"),
        "primary_headroom_fraction": round(
            (buckets.get("medium", 0) + buckets.get("hard", 0)) / n, 4),
    }
    assert "torch" not in sys.modules, "hardness analysis must not import torch"
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"[hardness] {name}: n={n} buckets={report['bucket_counts']} "
          f"medium+hard={report['primary_headroom_fraction']} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
