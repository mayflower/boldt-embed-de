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
    args = ap.parse_args()

    rows = _load_rows(args.input)
    summary = T.summarize_cache(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
        print(f"saved summary: {args.output}")

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
