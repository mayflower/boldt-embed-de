#!/usr/bin/env python3
"""Summarize (and optionally filter) a teacher cache — tells you if the data is usable before
training. Pure stdlib, no ML. Accepts a single cache JSONL or a shard manifest/glob.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import teacher as T  # noqa: E402  (stdlib at import time)


def _load_rows(input_path):
    p = pathlib.Path(input_path)
    if p.suffix == ".json" and p.name.endswith(".manifest.json"):
        man = json.loads(p.read_text(encoding="utf-8"))
        rows = []
        for sh in man.get("shards", []):
            sp = pathlib.Path(sh["path"])
            if sp.exists():
                rows += T.read_teacher_cache_jsonl(sp)
        return rows
    if any(ch in str(input_path) for ch in "*?["):
        rows = []
        for f in sorted(glob.glob(str(input_path))):
            rows += T.read_teacher_cache_jsonl(f)
        return rows
    return T.read_teacher_cache_jsonl(input_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="cache JSONL, *.manifest.json, or glob")
    ap.add_argument("--output", default=None, help="write summary JSON here")
    ap.add_argument("--filter-output", default=None, help="write training-kept cache here")
    ap.add_argument("--review-output", default=None, help="write below-threshold positives here")
    ap.add_argument("--reranker-threshold", type=float, default=0.0)
    ap.add_argument("--fail-on-unknown-license", action="store_true",
                    help="exit non-zero if any row has an unknown/missing license")
    ap.add_argument("--fail-on-disallowed-training-source", action="store_true",
                    help="exit non-zero if any row has allowed_for_training=false")
    ap.add_argument("--by-threshold", action="store_true",
                    help="also print the acceptance threshold sweep (see calibrate_teacher_thresholds.py)")
    args = ap.parse_args()

    rows = _load_rows(args.input)
    summary = T.summarize_cache(rows)
    if args.by_threshold:
        from boldt_embed import teacher_calibration as tc
        summary["acceptance_by_threshold"] = tc.acceptance_by_threshold(rows)
        summary["acceptance_by_threshold_by_domain"] = tc.acceptance_by_threshold_grouped(rows, "domain")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
        print(f"saved summary: {args.output}")

    # Hard failure modes (provenance gates) — checked AFTER the summary is written so the
    # report still lands, then a non-zero exit blocks any downstream training/release step.
    blockers = []
    if args.fail_on_unknown_license and summary["unknown_license_rows"] > 0:
        blockers.append(f"{summary['unknown_license_rows']} rows with unknown/missing license")
    if args.fail_on_disallowed_training_source and summary["disallowed_for_training_rows"] > 0:
        blockers.append(f"{summary['disallowed_for_training_rows']} rows with "
                        "allowed_for_training=false")
    if blockers:
        print("FAIL — teacher cache is not training-clean: " + "; ".join(blockers), file=sys.stderr)
        return 1

    if args.filter_output:
        split = T.filter_cache(rows, args.reranker_threshold)
        T.write_teacher_cache_jsonl(args.filter_output, split["kept"])
        print(f"[filter] kept {len(split['kept'])} -> {args.filter_output} "
              f"(reranker_threshold={args.reranker_threshold})")
        if args.review_output:
            T.write_teacher_cache_jsonl(args.review_output, split["review"])
            print(f"[filter] review {len(split['review'])} -> {args.review_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
