#!/usr/bin/env python3
"""Improved causal training: warmup -> GPU hard-negative mining (ANCE-style) -> continue
training with hard negatives. Evaluated base/warmup/final on BOTH a same-domain held-out set
(DT-de-dpr test) and a disjoint cross-domain benchmark (GerDaLIR legal). Requires extras + GPU.
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

from boldt_embed.config import load_causal_config  # noqa: E402
from boldt_embed.instructions import format_query  # noqa: E402


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load_dt(cfg, max_pairs, seed, log):
    from datasets import load_dataset

    ds = load_dataset("deutsche-telekom/wikipedia-22-12-de-dpr")
    train, test = ds["train"], ds["test"]
    ctx_index, corpus = {}, []
    pairs = []  # (query, positive_ctx, pos_idx)
    for row in train:
        ctx = row["context"]
        if ctx not in ctx_index:
            ctx_index[ctx] = len(corpus)
            corpus.append(ctx)
        ci = ctx_index[ctx]
        qs = list(row.get("question") or []) + list(row.get("imperative_informal") or []) \
            + list(row.get("imperative_formal") or [])
        for q in dict.fromkeys(x for x in qs if x and x.strip()):
            pairs.append((format_query(cfg.query_instruction, q), ctx, ci))
    random.Random(seed).shuffle(pairs)
    if max_pairs:
        pairs = pairs[:max_pairs]
    # same-domain held-out eval from the test split (one question per context)
    t_index, t_corpus, t_queries = {}, [], []
    for row in test:
        ctx = row["context"]
        if ctx not in t_index:
            t_index[ctx] = f"t{len(t_corpus)}"
            t_corpus.append({"id": t_index[ctx], "text": ctx})
        qs = list(row.get("question") or [])
        if qs:
            t_queries.append({"query": format_query(cfg.query_instruction, qs[0]),
                              "positive_ids": {t_index[ctx]}})
    log(f"[dt] train_pairs={len(pairs)} corpus={len(corpus)} dt_test(corpus={len(t_corpus)},q={len(t_queries)})")
    return pairs, corpus, t_corpus, t_queries


def load_gerdalir(cfg, log):
    from datasets import load_dataset

    c = load_dataset("mteb/GerDaLIRSmall", "corpus")["corpus"]
    q = load_dataset("mteb/GerDaLIRSmall", "queries")["queries"]
    rel = load_dataset("mteb/GerDaLIRSmall", "default")["test"]
    qrels = {}
    for r in rel:
        if float(r["score"]) > 0:
            qrels.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
    corpus = [{"id": str(r["_id"]), "text": ((r.get("title") or "") + " " + r["text"]).strip()} for r in c]
    queries = [{"query": format_query(cfg.query_instruction, x["text"]), "positive_ids": qrels[str(x["_id"])]}
               for x in q if str(x["_id"]) in qrels]
    log(f"[gerdalir] corpus={len(corpus)} queries={len(queries)}")
    return corpus, queries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "training_causal.json"))
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=150000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "checkpoints"))
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
        from boldt_embed import train as T
    except ImportError as exc:
        raise SystemExit(f"Needs extras: pip install -e '.[train]' + datasets. ({exc})")

    cfg = load_causal_config(args.config)
    pooling = "mean" if cfg.pooling == "mean" else "eos"
    di = args.device_index
    pairs, corpus, dt_corpus, dt_queries = load_dt(cfg, args.max_pairs, 0, print)
    ge_corpus, ge_queries = load_gerdalir(cfg, print)
    warm_dir = f"{args.out}/causal-hn-warmup"
    final_dir = f"{args.out}/causal-hn-final"

    def ev(path, tag):
        ge = T.retrieval_eval_real(path, ge_corpus, ge_queries, pooling=pooling, device_index=di, max_len=args.max_len)
        dt = T.retrieval_eval_real(path, dt_corpus, dt_queries, pooling=pooling, device_index=di, max_len=args.max_len)
        print(f"[eval:{tag}] gerdalir nDCG@10={ge['ndcg@10']:.4f}  dt_test nDCG@10={dt['ndcg@10']:.4f}")
        return {"gerdalir": ge, "dt_test": dt}

    print("=== eval BASE ===")
    base = ev(cfg.model_name_or_path, "base")

    print("=== WARMUP train (in-batch negatives, 1 epoch) ===")
    warm = T.train_pairs_real(cfg, [(q, p) for q, p, _ in pairs], output_dir=warm_dir,
                              device_index=di, epochs=1, batch_size=args.batch_size,
                              max_len=args.max_len, pooling=pooling, temperature=cfg.temperature)
    warm_eval = ev(warm_dir, "warmup")

    print("=== MINE hard negatives with warmup model ===")
    qtexts = [q for q, _, _ in pairs]
    pos_idx = [ci for _, _, ci in pairs]
    hn_idx = T.mine_hard_negatives_gpu(warm_dir, qtexts, corpus, pos_idx, k=1,
                                       pooling=pooling, device_index=di, max_len=args.max_len)
    triples = [(pairs[i][0], pairs[i][1], corpus[hn_idx[i]]) for i in range(len(pairs))]

    print("=== HARD-NEG train (continue from warmup, +mined negatives, 1 epoch) ===")
    final = T.train_triples_real(cfg, triples, output_dir=final_dir, device_index=di, epochs=1,
                                 batch_size=args.batch_size, max_len=args.max_len, pooling=pooling,
                                 temperature=cfg.temperature, init_from=warm_dir)
    final_eval = ev(final_dir, "final")

    report = {
        "status": "ok",
        "setup": "warmup(in-batch) -> GPU hard-neg mining -> continue w/ hard negs; "
                 "train=DT-de-dpr (Wikipedia); eval=DT-test (same-domain) + GerDaLIR (cross-domain legal)",
        "train_pairs": len(pairs), "mining": "embedding top-1 (excl positive), warmup model",
        "run_metadata": {"command": "scripts/train_hardneg_de.py", "commit": _git_commit(),
                         "date": "2026-05-29", "hardware": platform.platform(),
                         "gpu": final.get("gpu_name"), "torch": __import__("torch").__version__},
        "warmup_training": warm, "hardneg_training": final,
        "eval": {"base": base, "warmup": warm_eval, "final": final_eval},
    }
    out = ROOT / "outputs" / "real-training" / "hardneg-de-report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== SUMMARY (nDCG@10) ===")
    for tag, e in (("base", base), ("warmup", warm_eval), ("final", final_eval)):
        print(f"  {tag:7s} gerdalir={e['gerdalir']['ndcg@10']:.4f}  dt_test={e['dt_test']['ndcg@10']:.4f}")
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
