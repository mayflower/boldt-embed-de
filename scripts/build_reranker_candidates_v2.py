#!/usr/bin/env python3
"""Build v2 reranker training data as CANDIDATE LISTS (distribution-aware), to fix the v1
reranker generalization failure (degraded GermanQuAD). Each query gets the positive (label 1)
plus teacher-filtered hard negatives (label 0) from MULTIPLE sources (BM25 + optional dense),
with teacher scores and source/domain tags. Pure stdlib; dense sources via precomputed
embeddings only. Reports distribution so mismatch is visible before training.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data_pipeline as dp  # noqa: E402
from boldt_embed import negative_mining_2026 as nm  # noqa: E402


def _build_corpus(positives, corpus_path):
    lookup = {}
    if corpus_path and pathlib.Path(corpus_path).exists():
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
    ap.add_argument("--candidates", required=True, help="positives JSONL (query_id,query,doc_id,document)")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--teacher-cache", default=None)
    ap.add_argument("--dense-embeddings", default=None, help="JSONL {id, embedding} for query+doc ids")
    ap.add_argument("--output", default=str(ROOT / "data" / "processed" / "reranker_train_v2.jsonl"))
    ap.add_argument("--negatives-per-query", type=int, default=8)
    ap.add_argument("--false-negative-margin", type=float, default=0.1)
    ap.add_argument("--first-stage-k", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not pathlib.Path(args.candidates).exists():
        print(f"ERROR: candidates not found: {args.candidates}", file=sys.stderr); return 2
    positives = [r for r in dp.stream_jsonl(args.candidates) if r.get("positive", True)]
    corpus_lookup = _build_corpus(positives, args.corpus)
    corpus = list(corpus_lookup.values())
    queries = [{"query_id": p["query_id"], "query": p["query"]} for p in positives]

    pools = [("bm25", nm.mine_bm25_candidates(queries, corpus, k=args.first_stage_k))]
    if args.dense_embeddings and pathlib.Path(args.dense_embeddings).exists():
        emb = {str(r["id"]): r["embedding"] for r in dp.stream_jsonl(args.dense_embeddings)}
        q_emb = {q["query_id"]: emb[q["query_id"]] for q in queries if q["query_id"] in emb}
        d_emb = [(c["id"], emb[c["id"]]) for c in corpus if c["id"] in emb]
        if q_emb and d_emb:
            pools.append(("student_dense", nm.mine_dense_candidates_from_embeddings(
                q_emb, d_emb, k=args.first_stage_k)))
    merged = nm.merge_candidate_pools(*pools)

    teacher_scores = {}
    if args.teacher_cache and pathlib.Path(args.teacher_cache).exists():
        teacher_scores = nm.load_teacher_scores(list(dp.stream_jsonl(args.teacher_cache)))
        print(f"[teacher] {len(teacher_scores)} cached scores")
    else:
        print("[teacher] no cache -> false-negative filtering disabled")

    rows, stats = nm.build_reranker_candidate_lists(
        positives, merged, corpus_lookup, teacher_scores,
        negatives_per_query=args.negatives_per_query, margin=args.false_negative_margin)
    print(f"[reranker-lists] {json.dumps(stats, ensure_ascii=False)}")

    if args.dry_run:
        print("=== DRY RUN: not writing. First row: ===")
        if rows:
            print(json.dumps(rows[0], ensure_ascii=False))
        return 0
    n = dp.write_jsonl(args.output, rows)
    print(f"[write] {n} candidate-list rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
