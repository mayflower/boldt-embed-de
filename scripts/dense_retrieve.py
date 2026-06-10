#!/usr/bin/env python3
"""GPU dense retrieval helper: encode a corpus + queries with a SentenceTransformer and return
top-k by cosine (matmul on GPU). Two modes:

* ``negs``      -> candidate rows {query_id, doc_id, query, document, positive:false} for each
                   retrieved doc that is NOT the query's known positive (genuine hard negatives
                   for reranker training / teacher scoring).
* ``shortlist`` -> {query_id, query, candidates:[{doc_id, document}], positive_ids} per query
                   (fixed first-stage shortlists for eval_reranker_lift).

Requires extras + GPU. Pure matmul ranking — usable at corpus scale.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data_pipeline as dp  # noqa: E402


def _load_corpus(path):
    corpus = []
    for r in dp.stream_jsonl(path):
        did = str(r.get("doc_id") or r.get("id"))
        corpus.append({"doc_id": did, "document": r.get("document") or r.get("text") or ""})
    # dedup by doc_id keep first
    seen, out = set(), []
    for c in corpus:
        if c["doc_id"] not in seen:
            seen.add(c["doc_id"]); out.append(c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedder", required=True)
    ap.add_argument("--corpus", required=True, help="JSONL with doc_id/id + document/text")
    ap.add_argument("--queries", required=True,
                    help="JSONL with query_id/id + query/text [+ positive_ids/doc_id]")
    ap.add_argument("--mode", choices=["negs", "shortlist"], required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--query-instruction", default="")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(f"Needs extras: pip install -e '.[eval]'. ({exc})")

    corpus = _load_corpus(args.corpus)
    queries = []
    for r in dp.stream_jsonl(args.queries):
        qid = str(r.get("query_id") or r.get("id"))
        pos = set(str(x) for x in (r.get("positive_ids") or ([r["doc_id"]] if r.get("doc_id") else [])))
        queries.append({"query_id": qid, "query": r.get("query") or r.get("text") or "",
                        "positive_ids": pos})
    if args.limit:
        queries = queries[:args.limit]
    print(f"[dense] corpus={len(corpus)} queries={len(queries)} mode={args.mode} k={args.k}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.embedder, device=dev)
    model.max_seq_length = args.max_length
    cap = args.max_length * 8
    c_emb = model.encode([c["document"][:cap] for c in corpus], batch_size=64,
                         normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    q_emb = model.encode([(args.query_instruction + q["query"]) for q in queries], batch_size=64,
                         normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    cids = [c["doc_id"] for c in corpus]
    ctext = {c["doc_id"]: c["document"] for c in corpus}
    topk = min(args.k + 5, len(cids))

    rows = []
    for start in range(0, q_emb.size(0), 256):
        sims = q_emb[start:start + 256] @ c_emb.t()
        idx = torch.topk(sims, topk, dim=1).indices.tolist()
        for j, cand in enumerate(idx):
            q = queries[start + j]
            cand_ids = [cids[k] for k in cand]
            if args.mode == "negs":
                negs = [c for c in cand_ids if c not in q["positive_ids"]][:args.k]
                for did in negs:
                    rows.append({"query_id": q["query_id"], "doc_id": did, "query": q["query"],
                                 "document": ctext[did], "positive": False, "source": "dense_neg"})
            else:
                cand_ids = cand_ids[:args.k]
                rows.append({"query_id": q["query_id"], "query": q["query"],
                             "candidates": [{"doc_id": c, "document": ctext[c]} for c in cand_ids],
                             "positive_ids": sorted(q["positive_ids"])})
    dp.write_jsonl(args.out, rows)
    print(f"[dense] wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
