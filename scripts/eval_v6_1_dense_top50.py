#!/usr/bin/env python3
"""Evaluate dense retrievers (dense-v6.1 candidate vs dense-v6 / BM25 / e5-base) for German RAG
first-stage retrieval. DENSE quality only — no reranker, no policy. Pure-stdlib metric core
(testable); retrieval/encoding is lazy ML.

Per (model, eval set): Recall@10/20/50/100/200, nDCG@10, MRR@10, missing-positive rate, oracle
nDCG@10; for Boldt dense models also Matryoshka dims 1024/512/256/128 (+ 256 retention) and encode
throughput. Writes outputs/v6-1-dense-top50/eval/<model>__<set>.json + a summary.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed.metrics import mrr_at_k, ndcg_at_k, recall_at_k  # noqa: E402

KS = (10, 20, 50, 100, 200)
DIMS = (1024, 512, 256, 128)
EVAL_SETS = {
    "webfaq": ("outputs/v4-rag-reranker/eval/webfaq/corpus.jsonl",
               "outputs/v4-rag-reranker/eval/webfaq/queries.jsonl",
               "outputs/v4-rag-reranker/eval/webfaq/qrels.jsonl", "primary"),
    "germanquad": ("data/processed/eval/germanquad_corpus.jsonl",
                   "data/processed/eval/germanquad_queries.jsonl",
                   "data/processed/eval/germanquad_qrels.jsonl", "guardrail"),
    "dt_test": ("data/processed/eval/dt_test_corpus.jsonl",
                "data/processed/eval/dt_test_queries.jsonl",
                "data/processed/eval/dt_test_qrels.jsonl", "guardrail"),
    # MIRACL (de) REDUCED corpus (all relevant + 300k sampled distractors from the 15.9M corpus):
    # a fair cross-model comparison, NOT the official full-corpus leaderboard number.
    "miracl_de_reduced": ("data/processed/eval/miracl_de_reduced/corpus.jsonl",
                          "data/processed/eval/miracl_de_reduced/queries.jsonl",
                          "data/processed/eval/miracl_de_reduced/qrels.jsonl", "primary"),
}


def _read(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]


def _qrels(path, queries):
    pos = {}
    for r in _read(path):
        try:
            rel = float(r.get("relevance", 1))
        except (TypeError, ValueError):
            rel = 1.0
        if rel > 0 and r.get("doc_id"):
            pos.setdefault(str(r["query_id"]), set()).add(r["doc_id"])
    for q in queries:
        ids = q.get("positive_doc_ids") or ([q["positive_doc_id"]] if q.get("positive_doc_id") else [])
        if ids:
            pos.setdefault(str(q["query_id"]), set()).update(ids)
    return pos


def eval_rankings(rankings, qrels, ks=KS):
    """Pure-stdlib retrieval metrics. ``rankings``: qid -> ranked doc_ids. ``qrels``: qid -> set(pos)."""
    rec = {k: [] for k in ks}
    ndcg, mrr, present_flags = [], [], []
    maxk = max(ks)
    n = missing = 0
    for qid, pos in qrels.items():
        if not pos:
            continue
        n += 1
        r = rankings.get(qid, [])
        for k in ks:
            rec[k].append(recall_at_k(r, pos, k))
        ndcg.append(ndcg_at_k(r, pos, 10))
        mrr.append(mrr_at_k(r, pos, 10))
        present = bool(pos & set(r[:maxk]))
        present_flags.append(1.0 if present else 0.0)
        if not present:
            missing += 1

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0
    out = {f"recall@{k}": mean(rec[k]) for k in ks}
    out.update({"ndcg@10": mean(ndcg), "mrr@10": mean(mrr),
                "missing_positive_rate": round(missing / n, 4) if n else 0.0,
                "oracle_ndcg@10": mean(present_flags), "n_queries": n})
    return out


# --------------------------------------------------------------------- ML retrieval (lazy)
def _topk_from_sims(sims, doc_ids, k):
    import torch
    idx = torch.topk(sims, k=min(k, len(doc_ids))).indices.tolist()
    return [doc_ids[j] for j in idx]


def dense_eval(model_path, corpus_rows, queries, qrels, *, dims=DIMS, query_prefix="",
               doc_prefix="", max_seq_length=256, matryoshka=True, trust_remote_code=False):
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_path, device="cuda" if torch.cuda.is_available() else "cpu",
                            trust_remote_code=trust_remote_code)
    m.max_seq_length = max_seq_length
    dids = [c["doc_id"] for c in corpus_rows]
    t0 = time.time()
    demb = m.encode([doc_prefix + c.get("text", "") for c in corpus_rows], batch_size=256,
                    normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    doc_sec = round(len(dids) / max(time.time() - t0, 1e-6), 1)
    t0 = time.time()
    qemb = m.encode([query_prefix + q.get("query", "") for q in queries], batch_size=256,
                    normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    q_sec = round(len(queries) / max(time.time() - t0, 1e-6), 1)
    full_dim = demb.shape[1]
    use_dims = [d for d in dims if d <= full_dim] if matryoshka else [full_dim]
    dt_full = torch.tensor(demb)
    qt_full = torch.tensor(qemb)
    by_dim = {}
    for d in use_dims:
        if d == full_dim:
            dd, qq = dt_full, qt_full
        else:                                   # Matryoshka: truncate + renormalize
            dd = torch.nn.functional.normalize(dt_full[:, :d], dim=1)
            qq = torch.nn.functional.normalize(qt_full[:, :d], dim=1)
        rankings = {}
        for i, q in enumerate(queries):
            rankings[str(q["query_id"])] = _topk_from_sims(qq[i] @ dd.T, dids, max(KS))
        by_dim[d] = eval_rankings(rankings, qrels)
    primary = by_dim[full_dim if full_dim in by_dim else use_dims[0]]
    out = dict(primary)
    out["embedding_dim"] = full_dim
    out["matryoshka"] = {f"dim_{d}": {"ndcg@10": by_dim[d]["ndcg@10"],
                                      "recall@50": by_dim[d]["recall@50"],
                                      "recall@100": by_dim[d]["recall@100"]} for d in by_dim}
    if matryoshka and 256 in by_dim and full_dim in by_dim and by_dim[full_dim]["ndcg@10"]:
        out["matryoshka_256_retention"] = round(by_dim[256]["ndcg@10"] / by_dim[full_dim]["ndcg@10"], 4)
    out["throughput"] = {"docs_per_sec": doc_sec, "queries_per_sec": q_sec}
    return out


def bm25_eval(corpus_rows, queries, qrels):
    from boldt_embed import bm25_index as BM
    idx = BM.build_bm25_index(corpus_rows, text_field="text", id_field="doc_id", fold_umlauts=True)
    rankings = {str(q["query_id"]): [d for d, _ in idx.search(q.get("query", ""), top_k=max(KS))]
                for q in queries}
    return eval_rankings(rankings, qrels)


MODEL_SPECS = {
    "dense-v6.2": {"path": "outputs/v6-2-dense-guardrail/checkpoints/boldt-dense-rag-v6-2", "kind": "dense"},
    "dense-v6.1": {"path": "outputs/v6-1-dense-top50/checkpoints/boldt-dense-rag-v6-1", "kind": "dense"},
    "dense-v6": {"path": "outputs/v6-dense-rag/checkpoints/boldt-dense-rag-v6", "kind": "dense"},
    "bm25": {"kind": "bm25"},
    "e5-base": {"path": "intfloat/multilingual-e5-base", "kind": "dense",
                "query_prefix": "query: ", "doc_prefix": "passage: ", "matryoshka": False},
    # similar-sized multilingual baselines (real measured comparison)
    "gte-multilingual-base": {"path": "Alibaba-NLP/gte-multilingual-base", "kind": "dense",
                              "matryoshka": False, "trust_remote_code": True},
    "bge-m3": {"path": "BAAI/bge-m3", "kind": "dense", "matryoshka": False},
    "e5-large": {"path": "intfloat/multilingual-e5-large", "kind": "dense",
                 "query_prefix": "query: ", "doc_prefix": "passage: ", "matryoshka": False},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="dense-v6.1,dense-v6,bm25,e5-base")
    ap.add_argument("--eval-sets", default="webfaq,germanquad,dt_test")
    ap.add_argument("--output-dir", default="outputs/v6-1-dense-top50/eval")
    ap.add_argument("--summary", default="outputs/v6-1-dense-top50/dense_eval_summary.json")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sets = [s.strip() for s in args.eval_sets.split(",") if s.strip()]
    out_dir = pathlib.Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for s in sets:
        corpus_p, queries_p, qrels_p, role = EVAL_SETS[s]
        corpus = _read(corpus_p); queries = _read(queries_p); qrels = _qrels(qrels_p, queries)
        for name in models:
            spec = MODEL_SPECS.get(name)
            if not spec:
                print(f"skip unknown model {name}", file=sys.stderr); continue
            if spec["kind"] == "dense":
                p = spec["path"]
                if not (pathlib.Path(p).exists() or "/" in p and not p.startswith("outputs")):
                    if not pathlib.Path(p).exists() and p.startswith("outputs"):
                        print(f"skip {name}: checkpoint {p} missing", file=sys.stderr); continue
                res = dense_eval(p, corpus, queries, qrels, query_prefix=spec.get("query_prefix", ""),
                                 doc_prefix=spec.get("doc_prefix", ""),
                                 matryoshka=spec.get("matryoshka", True),
                                 trust_remote_code=spec.get("trust_remote_code", False))
            else:
                res = bm25_eval(corpus, queries, qrels)
            res["model"], res["eval_set"], res["role"] = name, s, role
            (out_dir / f"{name}__{s}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
            summary.setdefault(name, {})[s] = res
            print(f"[v6.1-dense-eval] {name} / {s} ({role}): R@50 {res.get('recall@50')} "
                  f"R@100 {res.get('recall@100')} nDCG@10 {res.get('ndcg@10')} "
                  f"missing {res.get('missing_positive_rate')} "
                  f"matryoshka256_ret {res.get('matryoshka_256_retention')}")
    pathlib.Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"[v6.1-dense-eval] summary -> {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
