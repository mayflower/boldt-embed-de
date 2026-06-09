#!/usr/bin/env python3
"""Build a leakage-aware, domain-balanced training-candidate JSONL.

Local-first: reads one or more ``--source-jsonl`` files of raw rows, normalizes them to the
candidate schema, optionally deduplicates, filters out anything that leaks into an eval
corpus, and balances per domain. Pure stdlib — no network, no ML deps.

Public benchmark *test* data (GermanQuAD / GerDaLIR / MMTEB) must be passed via
``--leakage-corpus-jsonl`` so any overlapping candidate is dropped: those datasets stay
eval-only.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data_pipeline as dp  # noqa: E402


def _load_eval_texts(paths):
    texts = []
    for p in paths or []:
        for row in dp.stream_jsonl(p):
            for f in ("query", "document", "text", "context", "title"):
                v = row.get(f)
                if isinstance(v, str) and v.strip():
                    texts.append(v)
    return texts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-jsonl", nargs="+", required=True,
                    help="one or more local JSONL files of raw rows")
    ap.add_argument("--output", default=str(ROOT / "data" / "processed" / "candidates.jsonl"))
    ap.add_argument("--default-source", default=None)
    ap.add_argument("--default-domain", default=None)
    ap.add_argument("--default-license", default=None)
    ap.add_argument("--max-per-domain", type=int, default=None)
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--leakage-corpus-jsonl", nargs="*", default=None,
                    help="eval corpus JSONL(s); candidates overlapping these texts are dropped")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw = []
    for src in args.source_jsonl:
        if not pathlib.Path(src).exists():
            print(f"ERROR: source not found: {src}", file=sys.stderr)
            return 2
        raw.extend(dp.stream_jsonl(src))
    rows = [dp.normalize_record(r, default_source=args.default_source,
                                default_domain=args.default_domain,
                                default_license=args.default_license) for r in raw]
    problems = []
    for i, r in enumerate(rows):
        for e in dp.validate_candidate_record(r):
            problems.append(f"row {i}: {e}")
    print(f"[normalize] {len(rows)} candidates; schema problems: {len(problems)}")
    for p in problems[:10]:
        print(f"  - {p}")
    if problems:
        print("ERROR: fix schema problems.", file=sys.stderr)
        return 2

    if args.dedup:
        before = len(rows)
        rows = dp.deduplicate_by_text_hash(rows)
        print(f"[dedup] {before} -> {len(rows)}")

    if args.leakage_corpus_jsonl:
        eval_texts = _load_eval_texts(args.leakage_corpus_jsonl)
        rows, stats = dp.filter_leakage_against_eval_texts(rows, eval_texts)
        print(f"[leakage] {stats}")

    if args.max_per_domain:
        before = len(rows)
        rows = dp.domain_balanced_sample(rows, args.max_per_domain)
        print(f"[balance] max_per_domain={args.max_per_domain}: {before} -> {len(rows)}")

    print(f"[domains] {dp.domain_counts(rows)}")
    if args.dry_run:
        print("=== DRY RUN: not writing. First 3 candidates: ===")
        for r in rows[:3]:
            print(json.dumps(r, ensure_ascii=False))
        return 0

    n = dp.write_jsonl(args.output, rows)
    print(f"[write] {n} candidates -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
