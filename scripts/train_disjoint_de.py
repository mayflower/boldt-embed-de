#!/usr/bin/env python3
"""At-scale, contamination-free German embedder training.

TRAIN on a NON-benchmark dataset (deutsche-telekom/wikipedia-22-12-de-dpr, German Wikipedia,
CC-BY-SA-4.0) using its question + formal/informal imperative variants as queries.
EVALUATE on a HELD-OUT, DISJOINT-DOMAIN benchmark (mteb/GerDaLIRSmall, German legal retrieval,
MIT) so the number measures cross-domain generalization, not in-domain memorization.

This is the honest replacement for the in-domain GermanQuAD run. Requires extras + GPU.
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


def build_train_pairs(cfg, max_pairs, seed, log):
    from datasets import load_dataset

    ds = load_dataset("deutsche-telekom/wikipedia-22-12-de-dpr")["train"]
    pairs = []
    for row in ds:
        ctx = row["context"]
        qs = list(row.get("question") or [])
        qs += list(row.get("imperative_informal") or [])
        qs += list(row.get("imperative_formal") or [])
        for q in dict.fromkeys(q for q in qs if q and q.strip()):  # dedup within row
            pairs.append((format_query(cfg.query_instruction, q), ctx))
    random.Random(seed).shuffle(pairs)
    if max_pairs:
        pairs = pairs[:max_pairs]
    log(f"[train-data] DT-de-dpr pairs={len(pairs)} (from {len(ds)} contexts)")
    return pairs


def build_gerdalir_eval(cfg, log):
    from datasets import load_dataset

    corpus_ds = load_dataset("mteb/GerDaLIRSmall", "corpus")["corpus"]
    queries_ds = load_dataset("mteb/GerDaLIRSmall", "queries")["queries"]
    qrels_ds = load_dataset("mteb/GerDaLIRSmall", "default")["test"]
    qrels = {}
    for r in qrels_ds:
        if float(r["score"]) > 0:
            qrels.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
    corpus = [{"id": str(r["_id"]), "text": ((r.get("title") or "") + " " + r["text"]).strip()}
              for r in corpus_ds]
    queries = [{"query": format_query(cfg.query_instruction, q["text"]),
                "positive_ids": qrels[str(q["_id"])]}
               for q in queries_ds if str(q["_id"]) in qrels]
    log(f"[eval-data] GerDaLIR corpus={len(corpus)} queries={len(queries)} (held-out legal)")
    return corpus, queries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "training_causal.json"))
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-pairs", type=int, default=300000)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "checkpoints" / "causal-disjoint-de"))
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
        from boldt_embed import train as T
    except ImportError as exc:
        raise SystemExit(f"Needs training extras: pip install -e '.[train]' + datasets. ({exc})")

    cfg = load_causal_config(args.config)
    pooling = "mean" if cfg.pooling == "mean" else "eos"
    pairs = build_train_pairs(cfg, args.max_pairs, args.seed, print)
    corpus, queries = build_gerdalir_eval(cfg, print)

    # Train (Wikipedia) and eval (legal) corpora are different sources -> no dedup needed,
    # but sanity-check leakage of a sample of training passages into the eval corpus.
    from boldt_embed.data import find_leakage
    sample = [{"positive": p[1]} for p in pairs[:2000]]
    leak = find_leakage(sample, [c["text"] for c in corpus][:2000], threshold=0.9)
    print(f"[leakage] train(2k)xeval(2k) sample hits: {len(leak)} (expect ~0; disjoint domains)")

    print("=== Eval BASE on held-out GerDaLIR (legal) ===")
    base = T.retrieval_eval_real(cfg.model_name_or_path, corpus, queries, pooling=pooling,
                                 device_index=args.device_index, max_len=args.max_len)
    print("base:", json.dumps(base))

    print("=== REAL at-scale training on DT-de-dpr (Wikipedia) ===")
    tr = T.train_pairs_real(cfg, pairs, output_dir=args.out, device_index=args.device_index,
                            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                            max_len=args.max_len, pooling=pooling, temperature=cfg.temperature)

    print("=== Eval TRAINED on held-out GerDaLIR (legal) ===")
    trained = T.retrieval_eval_real(args.out, corpus, queries, pooling=pooling,
                                    device_index=args.device_index, max_len=args.max_len)
    print("trained:", json.dumps(trained))

    report = {
        "status": "ok",
        "setup": "train=deutsche-telekom/wikipedia-22-12-de-dpr (Wikipedia, CC-BY-SA-4.0); "
                 "eval=mteb/GerDaLIRSmall (legal, MIT) — DISJOINT domains, held-out",
        "train_pairs": tr["num_pairs"], "train_steps": tr["steps"],
        "eval_corpus": len(corpus), "eval_queries": len(queries),
        "leakage_sample_hits": len(leak),
        "run_metadata": {
            "command": "scripts/train_disjoint_de.py", "commit": _git_commit(),
            "date": "2026-05-29", "hardware": platform.platform(),
            "gpu": tr.get("gpu_name"), "torch": __import__("torch").__version__,
        },
        "training": tr, "eval_base_model": base, "eval_trained_model": trained,
    }
    out_dir = ROOT / "outputs" / "real-training"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "disjoint-de-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== SUMMARY (held-out cross-domain GerDaLIR retrieval) ===")
    print(json.dumps({"base": base, "trained": trained, "train_pairs": tr["num_pairs"],
                      "steps": tr["steps"], "wall_time_sec": tr["wall_time_sec"]}, indent=2))
    print("saved:", out_dir / "disjoint-de-report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
