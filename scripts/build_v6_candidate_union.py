#!/usr/bin/env python3
"""Build v6 candidate UNION lists = dense-Boldt-v6 ∪ BM25, fused by Reciprocal Rank Fusion.

These are the inputs the v6 raw reranker trains/evals on: a candidate list per query where positives
are actually present (the v6 dense retriever fixed first-stage recall). Each candidate carries the
retriever source(s), the fused first-stage rank, and a gold label. Teacher scoring (Qwen3-Reranker-8B)
is a SEPARATE downstream step — this script only builds the union lists. The metric core (`rrf_fuse`)
is pure stdlib; dense encoding is lazy ML.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import bm25_index as BM  # noqa: E402
from boldt_embed.metrics import ndcg_at_k, recall_at_k  # noqa: E402


def _read(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]


def rrf_fuse(bm25_ranked, dense_ranked, *, rrf_k: int = 60, list_size: int = 100):
    """Reciprocal Rank Fusion of two ranked doc-id lists. Returns (fused_ids, score, sources).
    Deterministic (ties broken by doc_id). Pure stdlib."""
    score: Dict[str, float] = {}
    sources: Dict[str, set] = {}
    for src, ranked in (("bm25", bm25_ranked), ("dense_v6", dense_ranked)):
        for rank, d in enumerate(ranked):
            score[d] = score.get(d, 0.0) + 1.0 / (rrf_k + rank + 1)
            sources.setdefault(d, set()).add(src)
    fused = sorted(score, key=lambda d: (-score[d], str(d)))[:list_size]
    return fused, score, sources


def _qrels(path, queries):
    pos: Dict[str, set] = {}
    if path:
        for r in _read(path):
            try:
                rel = float(r.get("relevance", 1))
            except (TypeError, ValueError):
                rel = 1.0
            if rel > 0 and r.get("doc_id"):
                pos.setdefault(str(r["query_id"]), set()).add(r["doc_id"])
    for q in queries:                                   # fall back / merge query-level positives
        ids = q.get("positive_doc_ids") or q.get("positive_ids") or []
        if ids:
            pos.setdefault(str(q["query_id"]), set()).update(ids)
    return pos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qrels", default=None)
    ap.add_argument("--dense-model", default="outputs/v6-dense-rag/checkpoints/boldt-dense-rag-v6")
    ap.add_argument("--domain", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--bm25-topk", type=int, default=100)
    ap.add_argument("--dense-topk", type=int, default=100)
    ap.add_argument("--list-size", type=int, default=100)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--max-queries", type=int, default=0, help="0 = all (deterministic head slice)")
    ap.add_argument("--max-seq-length", type=int, default=256)
    args = ap.parse_args()

    corpus = _read(args.corpus)
    queries = _read(args.queries)
    if args.max_queries:
        queries = queries[:args.max_queries]
    qrels = _qrels(args.qrels, queries)
    doc_ids = [c["doc_id"] for c in corpus]
    doc_text = {c["doc_id"]: c.get("text", "") for c in corpus}

    # BM25 over the corpus
    idx = BM.build_bm25_index(corpus, text_field="text", id_field="doc_id", fold_umlauts=True)
    bm25 = {str(q["query_id"]): [d for d, _ in idx.search(q["query"], top_k=args.bm25_topk)]
            for q in queries}

    # dense-v6 over the corpus (lazy ML)
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(args.dense_model, device="cuda" if torch.cuda.is_available() else "cpu")
    m.max_seq_length = args.max_seq_length
    demb = m.encode([doc_text[d] for d in doc_ids], batch_size=256, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False)
    qemb = m.encode([q["query"] for q in queries], batch_size=256, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False)
    dt = torch.tensor(demb)

    rows: List[dict] = []
    present = 0
    rec = {f"recall@{k}": 0.0 for k in (10, 50, args.list_size)}
    ndcg = 0.0
    src_gold = {"bm25_only": 0, "dense_only": 0, "both": 0, "absent": 0}
    for qi, q in enumerate(queries):
        qid = str(q["query_id"])
        sims = torch.tensor(qemb[qi]) @ dt.T
        dtop = [doc_ids[j] for j in torch.topk(sims, k=min(args.dense_topk, len(doc_ids))).indices.tolist()]
        fused, score, sources = rrf_fuse(bm25.get(qid, []), dtop, rrf_k=args.rrf_k,
                                         list_size=args.list_size)
        pos = qrels.get(qid, set())
        cands = []
        for rank, d in enumerate(fused):
            s = sources[d]
            csrc = "bm25+dense_v6" if len(s) == 2 else next(iter(s))
            is_gold = d in pos
            cands.append({"doc_id": d, "text": doc_text.get(d, ""), "candidate_source": csrc,
                          "first_stage_rank": rank, "first_stage_score": round(score[d], 6),
                          "label": 1 if is_gold else None,
                          "high_precision_positive": bool(is_gold)})
        rows.append({"query_id": qid, "query": q.get("query", ""),
                     "positive_doc_ids": sorted(pos), "domain": args.domain or args.eval_set,
                     "candidates": cands})
        # metrics over the fused first-stage order
        for k in (10, 50, args.list_size):
            rec[f"recall@{k}"] += recall_at_k(fused, pos, k) if pos else 0.0
        ndcg += ndcg_at_k(fused, pos, 10) if pos else 0.0
        if pos:
            present += 1 if (pos & set(fused)) else 0
            for p in pos:
                if p in sources:
                    s = sources[p]
                    src_gold["both" if len(s) == 2 else ("bm25_only" if "bm25" in s else "dense_only")] += 1
                else:
                    src_gold["absent"] += 1
    n = len(queries) or 1
    report = {
        "eval_set": args.eval_set, "n_queries": len(queries), "corpus_docs": len(corpus),
        "list_size": args.list_size, "rrf_k": args.rrf_k,
        "positive_present_rate": round(present / n, 4),
        "union_recall": {k: round(v / n, 4) for k, v in rec.items()},
        "union_ndcg@10": round(ndcg / n, 4),
        "gold_source_contribution": src_gold,
    }
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.report:
        pathlib.Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    assert "torch" in sys.modules  # this path is the real ML build
    print(f"[v6-union] {args.eval_set}: {len(rows)} lists, present_rate {report['positive_present_rate']}, "
          f"union_recall {report['union_recall']}, gold_src {src_gold} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
