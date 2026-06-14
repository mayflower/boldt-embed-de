#!/usr/bin/env python3
"""Mine domain-balanced, teacher-filtered hard negatives for the 2026 workflow.

Multi-source candidate pools (BM25 always; dense if precomputed embeddings are supplied)
are merged, then teacher scores from the cache are used to drop likely false negatives and
keep the hardest plausible ones, balanced across domains. Pure stdlib — no network, no ML.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data_pipeline as dp  # noqa: E402
from boldt_embed import negative_mining_2026 as nm  # noqa: E402
from boldt_embed.bm25_index import BM25Index, build_bm25_index  # noqa: E402


def _build_corpus(positives, corpus_path):
    """Corpus = explicit --corpus file, else the union of candidate documents."""
    lookup = {}
    if corpus_path:
        for r in dp.stream_jsonl(corpus_path):
            did = str(r.get("doc_id") or r.get("id"))
            lookup[did] = {"id": did, "text": r.get("document") or r.get("text") or "",
                           "domain": r.get("domain", "unknown")}
    for p in positives:
        did = str(p["doc_id"])
        lookup.setdefault(did, {"id": did, "text": p.get("document", ""),
                                "domain": p.get("domain", "unknown")})
    return lookup


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", required=True, help="positive (query, doc) candidates JSONL")
    ap.add_argument("--corpus", default=None, help="optional negative-pool corpus JSONL")
    ap.add_argument("--teacher-cache", default=None, help="optional teacher-cache JSONL")
    ap.add_argument("--dense-embeddings", default=None,
                    help="optional JSONL of {id, embedding} for query AND doc ids")
    ap.add_argument("--output", default=str(ROOT / "data" / "processed" / "hard_negatives.jsonl"))
    ap.add_argument("--negatives-per-query", type=int, default=8)
    ap.add_argument("--false-negative-margin", type=float, default=0.1)
    ap.add_argument("--max-per-domain", type=int, default=None)
    ap.add_argument("--first-stage-k", type=int, default=50)
    ap.add_argument("--bm25-index", default=None,
                    help="prebuilt BM25 index JSON (build_bm25_index.py); else built once here")
    ap.add_argument("--max-queries", type=int, default=None,
                    help="explicitly cap the number of mined queries (subsample) — sets "
                         "mining_cap_applied=true")
    ap.add_argument("--require-full-corpus", action="store_true",
                    help="fail if mining would be capped/subsampled (no --max-queries)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not pathlib.Path(args.candidates).exists():
        print(f"ERROR: candidates not found: {args.candidates}", file=sys.stderr)
        return 2
    positives_all = [r for r in dp.stream_jsonl(args.candidates) if r.get("positive", True)]
    # corpus is the FULL negative pool regardless of any query cap.
    corpus_lookup = _build_corpus(positives_all, args.corpus)
    corpus = list(corpus_lookup.values())

    cap = args.max_queries
    mining_cap_applied = bool(cap is not None and cap < len(positives_all))
    if mining_cap_applied and args.require_full_corpus:
        print("ERROR: --require-full-corpus set but --max-queries caps mining to "
              f"{cap}/{len(positives_all)} queries.", file=sys.stderr)
        return 2
    positives = positives_all[:cap] if cap is not None else positives_all
    queries = [{"query_id": p["query_id"], "query": p["query"]} for p in positives]
    print(f"[mining] corpus={len(corpus)} queries={len(queries)}/{len(positives_all)} "
          f"cap_applied={mining_cap_applied}")

    t0 = time.monotonic()
    if args.bm25_index and pathlib.Path(args.bm25_index).exists():
        bm25 = BM25Index.load(args.bm25_index)
        print(f"[bm25] loaded prebuilt index: {bm25.n_docs} docs")
    else:
        bm25 = build_bm25_index(corpus)   # built ONCE, not per query
        print(f"[bm25] built index once: {bm25.n_docs} docs, {len(bm25.postings)} terms")
    pools = [("bm25", nm.mine_bm25_candidates(queries, corpus, k=args.first_stage_k, index=bm25))]
    bm25_runtime = round(time.monotonic() - t0, 4)
    if args.dense_embeddings:
        emb = {str(r["id"]): r["embedding"] for r in dp.stream_jsonl(args.dense_embeddings)}
        q_emb = {q["query_id"]: emb[q["query_id"]] for q in queries if q["query_id"] in emb}
        d_emb = [(c["id"], emb[c["id"]]) for c in corpus if c["id"] in emb]
        if q_emb and d_emb:
            pools.append(("dense", nm.mine_dense_candidates_from_embeddings(
                q_emb, d_emb, k=args.first_stage_k)))
            print(f"[dense] {len(q_emb)} query / {len(d_emb)} doc embeddings")
    merged = nm.merge_candidate_pools(*pools)

    teacher_scores = {}
    if args.teacher_cache and pathlib.Path(args.teacher_cache).exists():
        teacher_scores = nm.load_teacher_scores(list(dp.stream_jsonl(args.teacher_cache)))
        print(f"[teacher] {len(teacher_scores)} cached scores")
    else:
        print("[teacher] no cache -> false-negative filtering disabled (all candidates kept)")

    rows, stats = nm.build_triplets_or_lists(
        positives, merged, corpus_lookup, teacher_scores,
        negatives_per_query=args.negatives_per_query, margin=args.false_negative_margin,
        max_per_domain=args.max_per_domain)
    stats.update({"mining_corpus_size": len(corpus), "mining_query_count": len(queries),
                  "mining_cap_applied": mining_cap_applied, "bm25_runtime_sec": bm25_runtime})
    print(f"[mining] {json.dumps(stats, ensure_ascii=False)}")

    if args.dry_run:
        print("=== DRY RUN: not writing. First 2 rows: ===")
        for r in rows[:2]:
            print(json.dumps(r, ensure_ascii=False))
        return 0

    n = dp.write_jsonl(args.output, rows)
    report_path = pathlib.Path(args.output).with_suffix(".mining_report.json")
    report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {n} hard-negative rows -> {args.output}; report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
