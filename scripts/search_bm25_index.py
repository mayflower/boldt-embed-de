#!/usr/bin/env python3
"""Batch-search a prebuilt BM25 index with a queries JSONL (stdlib). The index is loaded once
and reused for every query (no rebuild). Writes one result row per query."""
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
    ap.add_argument("--index", required=True, help="BM25 index JSON (build_bm25_index.py)")
    ap.add_argument("--queries", required=True, help="queries JSONL ({query_id, query})")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    index = BM25Index.load(args.index)
    queries = [json.loads(l) for l in pathlib.Path(args.queries).read_text("utf-8").splitlines()
               if l.strip()]
    t0 = time.monotonic()
    results = index.batch_search([q.get("query", "") for q in queries], args.top_k)
    runtime = round(time.monotonic() - t0, 4)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for q, res in zip(queries, results):
            f.write(json.dumps({"query_id": q.get("query_id"), "query": q.get("query"),
                                "results": [{"doc_id": d, "score": s} for d, s in res]},
                               ensure_ascii=False) + "\n")
    print(f"[bm25] {len(queries)} queries x {index.n_docs} docs (top-{args.top_k}) "
          f"in {runtime}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
