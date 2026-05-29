#!/usr/bin/env python3
"""Real-scale German cross-encoder reranker: train on DT-de-dpr positives + embedder-mined
hard negatives, then measure reranking LIFT over an e5 first stage on held-out GerDaLIR.

Replaces the toy 7-pair reranker. Requires extras + GPU. Mining uses a trained embedder
(default: outputs/checkpoints/causal-hn-final if present, else the base model).
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

from boldt_embed.config import load_reranker_config  # noqa: E402
from boldt_embed.instructions import format_query  # noqa: E402
from boldt_embed.metrics import aggregate, metrics_for_query  # noqa: E402


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def build_reranker_examples(n_pairs, miner_model, device_index, log):
    """positives (q, ctx, 1) + one mined hard negative (q, hardneg_ctx, 0) per pair."""
    from datasets import load_dataset

    from boldt_embed import train as T

    ds = load_dataset("deutsche-telekom/wikipedia-22-12-de-dpr")["train"]
    ctx_index, corpus, raw = {}, [], []
    for row in ds:
        ctx = row["context"]
        if ctx not in ctx_index:
            ctx_index[ctx] = len(corpus)
            corpus.append(ctx)
        qs = list(row.get("question") or []) + list(row.get("imperative_formal") or [])
        for q in dict.fromkeys(x for x in qs if x and x.strip()):
            raw.append((q, ctx_index[ctx]))
    random.Random(0).shuffle(raw)
    raw = raw[:n_pairs]
    qtexts = [q for q, _ in raw]
    pos_idx = [ci for _, ci in raw]
    log(f"[rr-data] pairs={len(raw)} corpus={len(corpus)} miner={miner_model}")
    hn_idx = T.mine_hard_negatives_gpu(miner_model, qtexts, corpus, pos_idx, k=1,
                                       pooling="eos", device_index=device_index, max_len=192)
    examples = []
    for i, (q, ci) in enumerate(raw):
        examples.append((q, corpus[ci], 1.0))
        examples.append((q, corpus[hn_idx[i]], 0.0))
    random.Random(1).shuffle(examples)
    return examples


def load_gerdalir():
    from datasets import load_dataset

    c = load_dataset("mteb/GerDaLIRSmall", "corpus")["corpus"]
    q = load_dataset("mteb/GerDaLIRSmall", "queries")["queries"]
    rel = load_dataset("mteb/GerDaLIRSmall", "default")["test"]
    qrels = {}
    for r in rel:
        if float(r["score"]) > 0:
            qrels.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
    corpus = [{"id": str(r["_id"]), "text": ((r.get("title") or "") + " " + r["text"]).strip()} for r in c]
    queries = [{"id": str(x["_id"]), "text": x["text"], "positive_ids": qrels[str(x["_id"])]}
               for x in q if str(x["_id"]) in qrels]
    return corpus, queries


def rerank_eval(reranker_path, cfg, corpus, queries, device_index, n_queries=1000, top_k=50, log=print):
    """First stage = multilingual-e5-base top_k; rerank with the cross-encoder. nDCG@10 before/after."""
    import torch
    from sentence_transformers import SentenceTransformer

    from boldt_embed import train as T

    queries = queries[:n_queries]
    e5 = SentenceTransformer("intfloat/multilingual-e5-base", device=f"cuda:{device_index}")
    c_emb = e5.encode(["passage: " + c["text"] for c in corpus], batch_size=64,
                      normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    q_emb = e5.encode(["query: " + q["text"] for q in queries], batch_size=64,
                      normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    cids = [c["id"] for c in corpus]
    ctext = {c["id"]: c["text"] for c in corpus}
    topi = torch.topk(q_emb @ c_emb.t(), min(top_k, len(cids)), dim=1).indices.tolist()

    base_rows, rr_rows = [], []
    pairs, layout = [], []
    for i, q in enumerate(queries):
        cand = [cids[j] for j in topi[i]]
        base_rows.append(metrics_for_query(cand, set(q["positive_ids"]), (10,)))
        layout.append((i, cand))
        for cid in cand:
            pairs.append((format_query(cfg.query_instruction, q["text"]), ctext[cid]))
    log(f"[rr-eval] scoring {len(pairs)} (query,doc) pairs with cross-encoder")
    scores = T.rerank_scores_batch(reranker_path, pairs, template=cfg.input_template,
                                   device_index=device_index, max_len=256)
    p = 0
    for i, cand in layout:
        s = scores[p:p + len(cand)]
        p += len(cand)
        reranked = [cid for cid, _ in sorted(zip(cand, s), key=lambda kv: kv[1], reverse=True)]
        rr_rows.append(metrics_for_query(reranked, set(queries[i]["positive_ids"]), (10,)))
    return {"first_stage_e5": aggregate(base_rows), "e5_plus_reranker": aggregate(rr_rows),
            "n_queries": len(queries), "top_k": top_k}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "training_reranker.json"))
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--miner-model", default=str(ROOT / "outputs" / "checkpoints" / "causal-hn-final"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "checkpoints" / "reranker-de"))
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
        from boldt_embed import train as T
    except ImportError as exc:
        raise SystemExit(f"Needs extras: pip install -e '.[train,eval]' + datasets. ({exc})")

    cfg = load_reranker_config(args.config)
    miner = args.miner_model if pathlib.Path(args.miner_model).exists() else cfg.model_name_or_path
    examples = build_reranker_examples(args.pairs, miner, args.device_index, print)
    print(f"=== Train reranker on {len(examples)} examples ===")
    tr = T.train_reranker_scaled(cfg, examples, output_dir=args.out, device_index=args.device_index,
                                 epochs=args.epochs, batch_size=args.batch_size)
    print("=== Reranking eval on GerDaLIR (e5 first stage vs +reranker) ===")
    corpus, queries = load_gerdalir()
    ev = rerank_eval(args.out, cfg, corpus, queries, args.device_index)

    report = {"status": "ok", "miner_model": miner,
              "setup": "train=DT-de-dpr pos + embedder-mined hard negs; eval=rerank e5 top-50 on GerDaLIR",
              "run_metadata": {"command": "scripts/train_reranker_de.py", "commit": _git_commit(),
                               "date": "2026-05-29", "hardware": platform.platform(),
                               "gpu": tr.get("gpu_name"), "torch": __import__("torch").__version__},
              "training": tr, "reranking_eval": ev}
    out = ROOT / "outputs" / "real-training" / "reranker-de-report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== SUMMARY (GerDaLIR rerank nDCG@10) ===")
    print(json.dumps({"e5_first_stage": ev["first_stage_e5"]["ndcg@10"],
                      "e5_plus_reranker": ev["e5_plus_reranker"]["ndcg@10"],
                      "examples": tr["num_examples"]}, indent=2))
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
