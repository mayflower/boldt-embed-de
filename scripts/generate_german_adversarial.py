#!/usr/bin/env python3
"""Generate German adversarial candidates from seed (query, document) pairs.

Deterministic, pure stdlib. Each seed yields the anchor pair plus orthographic/register/
legal-wording paraphrases (positive) and negation/number/date/legal/entity distractors
(hard negatives), all marked source=synthetic_adversarial, domain=german_stress.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data_pipeline as dp  # noqa: E402
from boldt_embed import german_adversarial as ga  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="seed JSONL with 'query' and 'document'")
    ap.add_argument("--output", default=str(ROOT / "data" / "processed" / "adversarial_candidates.jsonl"))
    ap.add_argument("--include", nargs="*", default=None,
                    help="restrict to these template_ids (default: all)")
    ap.add_argument("--no-anchor", action="store_true", help="do not emit the original pair")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not pathlib.Path(args.input).exists():
        print(f"ERROR: seed file not found: {args.input}", file=sys.stderr)
        return 2
    seeds = list(dp.stream_jsonl(args.input))
    rows = ga.generate_adversarial_candidates(seeds, include=args.include,
                                              emit_anchor=not args.no_anchor)
    pos = sum(1 for r in rows if r["positive"])
    print(f"[adversarial] {len(seeds)} seeds -> {len(rows)} candidates "
          f"({pos} positive / {len(rows) - pos} distractor)")
    by_template = {}
    for r in rows:
        t = r["metadata"]["template_id"]
        by_template[t] = by_template.get(t, 0) + 1
    print(f"[templates] {dict(sorted(by_template.items()))}")

    if args.dry_run:
        print("=== DRY RUN: not writing. First 4 candidates: ===")
        for r in rows[:4]:
            print(json.dumps(r, ensure_ascii=False))
        return 0

    n = dp.write_jsonl(args.output, rows)
    print(f"[write] {n} candidates -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
