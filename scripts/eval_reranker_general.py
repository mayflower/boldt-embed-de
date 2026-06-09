#!/usr/bin/env python3
"""GENERAL reranking eval (not legal). Reranks BM25 and e5 first-stages on general German
retrieval (GermanQuAD test + held-out DT-de-dpr test) with the trained cross-encoder, and
reports nDCG@10 before/after. A good general reranker should LIFT, especially over BM25.

Eval-only against an existing reranker checkpoint. Requires extras + GPU.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import random
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data as datamod  # noqa: E402
from boldt_embed.config import load_reranker_config  # noqa: E402
from boldt_embed.eval_harness import bm25_rank  # noqa: E402
from boldt_embed.metrics import aggregate, metrics_for_query  # noqa: E402


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def germanquad_eval(n_q):
    from datasets import load_dataset
    base = ("https://huggingface.co/datasets/deepset/germanquad/resolve/"
            "refs%2Fconvert%2Fparquet/plain_text")
    test = load_dataset("parquet", data_files={"test": f"{base}/test/0000.parquet"})["test"]
    cidx, corpus, queries = {}, [], []
    for ex in test:
        c = ex["context"]
        if c not in cidx:
            cidx[c] = f"g{len(corpus)}"
            corpus.append({"id": cidx[c], "text": c})
        queries.append({"text": ex["question"], "positive_ids": {cidx[c]}})
    random.Random(0).shuffle(queries)
    return corpus, queries[:n_q]


def dt_test_eval(n_q):
    from datasets import load_dataset
    test = load_dataset("deutsche-telekom/wikipedia-22-12-de-dpr")["test"]
    cidx, corpus, queries = {}, [], []
    for row in test:
        c = row["context"]
        if c not in cidx:
            cidx[c] = f"d{len(corpus)}"
            corpus.append({"id": cidx[c], "text": c})
        qs = list(row.get("question") or [])
        if qs:
            queries.append({"text": qs[0], "positive_ids": {cidx[c]}})
    random.Random(0).shuffle(queries)
    return corpus, queries[:n_q]


def first_stage_e5(corpus, queries, top_k, device_index):
    import torch
    from sentence_transformers import SentenceTransformer
    e5 = SentenceTransformer("intfloat/multilingual-e5-base", device=f"cuda:{device_index}")
    C = e5.encode(["passage: " + c["text"] for c in corpus], batch_size=64,
                  normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    Q = e5.encode(["query: " + q["text"] for q in queries], batch_size=64,
                  normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    cids = [c["id"] for c in corpus]
    topi = torch.topk(Q @ C.t(), min(top_k, len(cids)), dim=1).indices.tolist()
    return [[cids[j] for j in row] for row in topi]


def first_stage_bm25(corpus, queries, top_k):
    out = []
    for q in queries:
        out.append(bm25_rank(q["text"], corpus)[:top_k])
    return out


def rerank_and_score(reranker_path, cfg, corpus, queries, shortlists, device_index):
    from boldt_embed import train as T
    ctext = {c["id"]: c["text"] for c in corpus}
    base_rows, pairs, layout = [], [], []
    for i, q in enumerate(queries):
        cand = shortlists[i]
        base_rows.append(metrics_for_query(cand, set(q["positive_ids"]), (10,)))
        layout.append((i, cand))
        for cid in cand:
            pairs.append((q["text"], ctext[cid]))
    scores = T.rerank_scores_batch(reranker_path, pairs, template=cfg.input_template,
                                   device_index=device_index, max_len=256)
    rr_rows, p = [], 0
    for i, cand in layout:
        s = scores[p:p + len(cand)]; p += len(cand)
        reranked = [cid for cid, _ in sorted(zip(cand, s), key=lambda kv: kv[1], reverse=True)]
        rr_rows.append(metrics_for_query(reranked, set(queries[i]["positive_ids"]), (10,)))
    return aggregate(base_rows)["ndcg@10"], aggregate(rr_rows)["ndcg@10"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reranker", default=str(ROOT / "outputs" / "checkpoints" / "reranker-de"))
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--gq-queries", type=int, default=1500)
    ap.add_argument("--dt-queries", type=int, default=1000)
    args = ap.parse_args()
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"Needs extras: pip install -e '.[train,eval]'. ({exc})")
    cfg = load_reranker_config(ROOT / "configs" / "training_reranker.json")

    results = {}
    for name, (corpus, queries) in {
        "germanquad_test": germanquad_eval(args.gq_queries),
        "dt_test": dt_test_eval(args.dt_queries),
    }.items():
        print(f"=== {name}: corpus={len(corpus)} queries={len(queries)} ===")
        block = {}
        # e5 first stage
        sl = first_stage_e5(corpus, queries, args.top_k, args.device_index)
        b, r = rerank_and_score(args.reranker, cfg, corpus, queries, sl, args.device_index)
        block["e5"] = {"first_stage": round(b, 4), "plus_reranker": round(r, 4)}
        print(f"  e5   {b:.4f} -> {r:.4f}")
        # BM25 first stage (general task; reranker should lift this clearly)
        sl = first_stage_bm25(corpus, queries, args.top_k)
        b, r = rerank_and_score(args.reranker, cfg, corpus, queries, sl, args.device_index)
        block["bm25"] = {"first_stage": round(b, 4), "plus_reranker": round(r, 4)}
        print(f"  bm25 {b:.4f} -> {r:.4f}")
        results[name] = block

    report = {"status": "ok", "reranker": args.reranker, "top_k": args.top_k,
              "note": "GENERAL reranking eval (GermanQuAD + DT-test); nDCG@10 first-stage vs +reranker",
              "run_metadata": {"command": "scripts/eval_reranker_general.py", "commit": _git_commit(),
                               "date": "2026-06-09", "hardware": platform.platform()},
              "results": results}
    out = ROOT / "outputs" / "real-training" / "reranker-general-report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== SUMMARY (nDCG@10 first-stage -> +reranker) ===")
    print(json.dumps(results, indent=2))
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
