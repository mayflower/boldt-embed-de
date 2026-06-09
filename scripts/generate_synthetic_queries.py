#!/usr/bin/env python3
"""Generate template-based German synthetic queries from passages.

Deterministic, pure stdlib, no network. Output is candidate rows (query→passage, positive)
in the standard schema; a later teacher pass scores/filters them before training.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data_pipeline as dp  # noqa: E402
from boldt_embed import synthetic_queries as sq  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--passages", required=True, help="JSONL of passages (text + domain + license)")
    ap.add_argument("--output", default=str(ROOT / "data" / "processed" / "synthetic_candidates.jsonl"))
    ap.add_argument("--queries-per-passage", type=int, default=None)
    ap.add_argument("--domains", nargs="*", default=None,
                    help=f"restrict query styles (choices: {sq.ALL_QUERY_STYLES})")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not pathlib.Path(args.passages).exists():
        print(f"ERROR: passages file not found: {args.passages}", file=sys.stderr)
        return 2
    passages = list(dp.stream_jsonl(args.passages))
    rows = sq.generate_synthetic_candidates(passages, args.queries_per_passage, args.domains)
    by_style = {}
    for r in rows:
        st = r["metadata"]["query_style"]
        by_style[st] = by_style.get(st, 0) + 1
    print(f"[synthetic] {len(passages)} passages -> {len(rows)} query candidates")
    print(f"[styles] {dict(sorted(by_style.items()))}")

    if args.dry_run:
        print("=== DRY RUN: not writing. First 5 candidates: ===")
        for r in rows[:5]:
            print(json.dumps(r, ensure_ascii=False))
        return 0

    n = dp.write_jsonl(args.output, rows)
    print(f"[write] {n} candidates -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
