#!/usr/bin/env python3
"""REAL causal-embedder training + evaluation on GermanQuAD (deepset/germanquad, CC-BY-4.0).

Trains on thousands of real German question/passage pairs (MNRL, in-batch negatives,
multi-epoch, bf16) and evaluates retrieval on the held-out TEST split (questions vs unique
test passages) with real nDCG/MRR/Recall — base model vs trained model.

Requires: pip install -e '.[train]' + datasets, and a GPU. Downloads ~tens of MB.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
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


def build_data(cfg, max_train, log):
    from datasets import load_dataset

    # The hub loader script is no longer supported; load the auto-converted parquet directly.
    base = ("https://huggingface.co/datasets/deepset/germanquad/resolve/"
            "refs%2Fconvert%2Fparquet/plain_text")
    ds = load_dataset("parquet", data_files={
        "train": f"{base}/train/0000.parquet", "test": f"{base}/test/0000.parquet"})
    train = ds["train"]
    test = ds["test"]

    # Train pairs: (instructed question, context). Dedup; cap for a bounded real run.
    seen = set()
    pairs = []
    for ex in train:
        key = (ex["question"], ex["context"])
        if key in seen:
            continue
        seen.add(key)
        pairs.append((format_query(cfg.query_instruction, ex["question"]), ex["context"]))
        if max_train and len(pairs) >= max_train:
            break

    # Eval: unique TEST contexts -> corpus; each test question -> its context as positive.
    ctx_to_id = {}
    corpus = []
    for ex in test:
        c = ex["context"]
        if c not in ctx_to_id:
            cid = f"c{len(corpus)}"
            ctx_to_id[c] = cid
            corpus.append({"id": cid, "text": c})
    queries = [{"query": format_query(cfg.query_instruction, ex["question"]),
                "positive_ids": [ctx_to_id[ex["context"]]]} for ex in test]
    log(f"[data] train_pairs={len(pairs)} test_corpus={len(corpus)} test_queries={len(queries)}")
    return pairs, corpus, queries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "training_causal.json"))
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-train", type=int, default=0, help="0 = use all train pairs")
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "checkpoints" / "causal-germanquad"))
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
        from boldt_embed import train as T
    except ImportError as exc:
        raise SystemExit(f"Needs training extras: pip install -e '.[train]' + datasets. ({exc})")

    cfg = load_causal_config(args.config)
    pooling = "mean" if cfg.pooling == "mean" else "eos"
    pairs, corpus, queries = build_data(cfg, args.max_train or None, print)

    print("=== Eval BASE model on GermanQuAD test retrieval ===")
    base = T.retrieval_eval_real(cfg.model_name_or_path, corpus, queries,
                                 pooling=pooling, device_index=args.device_index,
                                 max_len=args.max_len)
    print("base:", json.dumps(base))

    print("=== REAL training on GermanQuAD train pairs ===")
    train_report = T.train_pairs_real(
        cfg, pairs, output_dir=args.out, device_index=args.device_index,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_len=args.max_len,
        pooling=pooling, temperature=cfg.temperature)

    print("=== Eval TRAINED model on GermanQuAD test retrieval ===")
    trained = T.retrieval_eval_real(args.out, corpus, queries, pooling=pooling,
                                    device_index=args.device_index, max_len=args.max_len)
    print("trained:", json.dumps(trained))

    report = {
        "status": "ok",
        "dataset": "deepset/germanquad",
        "dataset_license": "cc-by-4.0",
        "task": "GermanQuAD test retrieval (questions vs unique test passages)",
        "run_metadata": {
            "command": "scripts/train_causal_germanquad.py", "commit": _git_commit(),
            "date": "2026-05-29", "hardware": platform.platform(),
            "gpu": train_report.get("gpu_name"), "torch": __import__("torch").__version__,
        },
        "train_pairs": train_report["num_pairs"],
        "test_corpus": len(corpus),
        "test_queries": len(queries),
        "training": train_report,
        "eval_base_model": base,
        "eval_trained_model": trained,
    }
    out_dir = ROOT / "outputs" / "real-training"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "germanquad-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== SUMMARY (real held-out GermanQuAD retrieval) ===")
    print(json.dumps({"base": base, "trained": trained,
                      "train_pairs": report["train_pairs"], "steps": train_report["steps"],
                      "wall_time_sec": train_report["wall_time_sec"]}, indent=2))
    print("saved:", out_dir / "germanquad-report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
