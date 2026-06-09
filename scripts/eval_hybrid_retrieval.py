#!/usr/bin/env python3
"""Hybrid retrieval evaluation: BM25 + dense + RRF + reranker, with a Matryoshka sweep (Prompt 8).

`--dry-run` validates the inputs and computes the **BM25-only** numbers (pure stdlib, no model
download) plus the planned modes/dims. The dense, hybrid, reranker, and Matryoshka paths need
the `eval` extras + a GPU.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data_pipeline as dp  # noqa: E402
from boldt_embed import experiment_registry as ER  # noqa: E402
from boldt_embed import hybrid_eval as H  # noqa: E402

MODES = ["bm25_only", "dense_only", "hybrid_rrf", "hybrid_rrf_plus_reranker"]


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _load_inputs(corpus_p, queries_p, qrels_p):
    corpus = [{"id": str(r.get("doc_id") or r.get("id")),
               "text": r.get("document") or r.get("text") or ""} for r in dp.stream_jsonl(corpus_p)]
    qrels = {}
    for r in dp.stream_jsonl(qrels_p):
        if float(r.get("relevance", r.get("score", 1))) > 0:
            qrels.setdefault(str(r["query_id"]), set()).add(str(r.get("doc_id") or r.get("corpus-id")))
    queries = []
    for r in dp.stream_jsonl(queries_p):
        qid = str(r.get("query_id") or r.get("id"))
        queries.append({"query_id": qid, "query": r.get("query") or r.get("text") or "",
                        "positive_ids": qrels.get(qid, set())})
    queries = [q for q in queries if q["positive_ids"]]
    return corpus, queries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedder-model", default=None)
    ap.add_argument("--reranker-model", default=None)
    ap.add_argument("--eval-corpus", required=True)
    ap.add_argument("--eval-queries", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--dims", default="1024,768,512,256,128,64")
    ap.add_argument("--top-k-first-stage", type=int, default=200)
    ap.add_argument("--top-k-rerank", type=int, default=50)
    ap.add_argument("--config", default=str(ROOT / "configs" / "training_reranker.json"))
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--output", default=str(ROOT / "outputs" / "eval" / "hybrid_eval.json"))
    ap.add_argument("--run-id", default=None, help="experiment run id (run card written on success)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for p in (args.eval_corpus, args.eval_queries, args.qrels):
        if not pathlib.Path(p).exists():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            return 2
    corpus, queries = _load_inputs(args.eval_corpus, args.eval_queries, args.qrels)
    dims = [int(x) for x in args.dims.split(",") if x.strip()]
    print(f"[inputs] corpus={len(corpus)} queries={len(queries)} dims={dims} modes={MODES}")

    bm25 = H.bm25_rankings_for_queries(queries, corpus)
    bm25_metrics = H.evaluate_mode(queries, bm25, {}, "bm25_only",
                                   top_k_first_stage=args.top_k_first_stage)
    print(f"[bm25_only] {json.dumps(bm25_metrics, ensure_ascii=False)}")

    if args.dry_run:
        assert "torch" not in sys.modules, "dry-run must not import torch"
        print(f"[plan] would also run: {MODES[1:]} + Matryoshka sweep over {dims}")
        print("dry-run-ok (no ML imports)")
        return 0

    if not args.embedder_model:
        print("ERROR: --embedder-model required for non-dry-run.", file=sys.stderr)
        return 2
    try:
        import torch  # noqa: F401
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(f"Needs extras: pip install -e '.[eval]'. ({exc})")

    t0 = time.time()
    device = f"cuda:{args.device_index}"
    enc = SentenceTransformer(args.embedder_model, device=device)
    c_emb = enc.encode([c["text"] for c in corpus], normalize_embeddings=True,
                       show_progress_bar=False).tolist()
    q_emb = enc.encode([q["query"] for q in queries], normalize_embeddings=True,
                       show_progress_bar=False).tolist()
    doc_vecs = [(corpus[i]["id"], c_emb[i]) for i in range(len(corpus))]
    query_vecs = {queries[i]["query_id"]: q_emb[i] for i in range(len(queries))}
    dense = {q["query_id"]: H.cosine_rank(query_vecs[q["query_id"]], doc_vecs) for q in queries}
    encode_secs = round(time.time() - t0, 2)

    rerank_fns = None
    if args.reranker_model:
        from boldt_embed import reranker_modern as RM
        from boldt_embed.config import load_reranker_config
        rcfg = load_reranker_config(args.config)
        ctext = {c["id"]: c["text"] for c in corpus}
        rerank_fns = {}
        for q in queries:
            qtext = q["query"]
            def _fn(head, qtext=qtext):
                scores = RM.score_with_student_reranker(
                    args.reranker_model, [(qtext, ctext[d]) for d in head],
                    rcfg.input_template, device=device)
                return [d for d, _ in sorted(zip(head, scores), key=lambda kv: kv[1], reverse=True)]
            rerank_fns[q["query_id"]] = _fn

    results = {}
    for mode in MODES:
        if mode == "hybrid_rrf_plus_reranker" and not rerank_fns:
            continue
        results[mode] = H.evaluate_mode(queries, bm25, dense, mode,
                                        top_k_first_stage=args.top_k_first_stage,
                                        top_k_rerank=args.top_k_rerank, rerank_fns=rerank_fns)
    sweep = H.matryoshka_sweep(query_vecs, doc_vecs, queries, dims)

    report = {"status": "ok", "n_queries": len(queries), "n_corpus": len(corpus),
              "modes": results, "matryoshka_sweep": {str(d): m for d, m in sweep.items()},
              "throughput": {"encode_seconds": encode_secs,
                             "texts_per_sec": round((len(corpus) + len(queries)) / max(encode_secs, 1e-6), 1)},
              "run_metadata": {"command": "scripts/eval_hybrid_retrieval.py", "commit": _git_commit(),
                               "hardware": platform.platform(), "embedder": args.embedder_model,
                               "reranker": args.reranker_model}}
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    card = ER.emit_run_card(args.run_id, "eval", "scripts/eval_hybrid_retrieval.py",
                            model=args.embedder_model, dataset=args.eval_corpus,
                            metrics={m: r.get("ndcg@10") for m, r in results.items()},
                            input_artifacts=[args.eval_corpus, args.eval_queries, args.qrels],
                            output_artifacts=[str(out)], notes="hybrid BM25+dense+RRF+reranker")
    print("=== SUMMARY (nDCG@10 by mode) ===")
    for mode, m in results.items():
        print(f"  {mode:28s} ndcg@10={m.get('ndcg@10')}")
    print(f"saved: {out}; run card: {card}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
