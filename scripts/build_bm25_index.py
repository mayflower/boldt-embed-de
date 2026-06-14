#!/usr/bin/env python3
"""Build a reusable BM25 inverted index over a corpus JSONL (stdlib, no ML, no network).

Build once here, then reuse via `search_bm25_index.py --index` or
`mine_hard_negatives_2026.py --bm25-index` — never rebuild per query.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed.bm25_index import BM25Index  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, help="corpus JSONL")
    ap.add_argument("--output", required=True, help="index JSON output path")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--id-field", default="doc_id")
    ap.add_argument("--fold-umlauts", action="store_true", help="also fold ä->ae, ö->oe, ü->ue")
    args = ap.parse_args()

    cp = pathlib.Path(args.corpus)
    if not cp.exists():
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        return 2
    docs = [json.loads(l) for l in cp.read_text(encoding="utf-8").splitlines() if l.strip()]
    t0 = time.monotonic()
    index = BM25Index(fold_umlauts=args.fold_umlauts).build(
        docs, text_field=args.text_field, id_field=args.id_field)
    runtime = round(time.monotonic() - t0, 4)
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    index.save(str(out))
    print(f"[bm25] indexed {index.n_docs} docs, {len(index.postings)} terms, "
          f"avgdl={index.avgdl:.1f} in {runtime}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
