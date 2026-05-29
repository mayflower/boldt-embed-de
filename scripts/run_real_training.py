#!/usr/bin/env python3
"""REAL training + evaluation of the causal embedder on GPU.

Requires the training extras and downloads the base weights:
    pip install -e '.[train]'   (or: pip install torch transformers accelerate safetensors)

Trains on the toy German triples, saves a fine-tuned checkpoint, then evaluates BOTH the
base model and the trained model on the toy retrieval benchmark with real embeddings.
Honest scale: this is a tiny real run that proves the pipeline trains and improves a real
model — it is NOT a production model or a public-benchmark claim.
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

from boldt_embed import data as datamod  # noqa: E402
from boldt_embed.config import load_causal_config  # noqa: E402
from boldt_embed.eval_harness import cosine_rank  # noqa: E402
from boldt_embed.instructions import format_document, format_query  # noqa: E402
from boldt_embed.metrics import aggregate, metrics_for_query  # noqa: E402

TRIPLES = ROOT / "data" / "samples" / "toy_triples_de.jsonl"
BENCH = ROOT / "benchmarks" / "toy_de_retrieval.json"
KS = (1, 3, 5, 10)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def evaluate_model(model_path: str, data: dict, cfg, pooling: str, device_index: int) -> dict:
    from boldt_embed import train as T

    corpus = data["corpus"]
    doc_ids = [d["id"] for d in corpus]
    doc_texts = [format_document(cfg.document_instruction, d["text"]) for d in corpus]
    q_texts = [format_query(cfg.query_instruction, q["query"]) for q in data["queries"]]
    doc_vecs = T.encode_texts(model_path, doc_texts, pooling=pooling, device_index=device_index)
    q_vecs = T.encode_texts(model_path, q_texts, pooling=pooling, device_index=device_index)

    rows, per = [], []
    for i, qd in enumerate(data["queries"]):
        ranked = cosine_rank(q_vecs[i], list(zip(doc_ids, doc_vecs)))
        pos = set(qd["positive_doc_ids"])
        m = metrics_for_query(ranked, pos, KS)
        rows.append(m)
        per.append({"query_id": qd["id"], "top5": ranked[:5],
                    "positive_doc_ids": sorted(pos), "ndcg@10": m["ndcg@10"]})
    return {"aggregate": aggregate(rows), "queries": per}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "training_causal.json"))
    parser.add_argument("--device-index", type=int, default=1, help="CUDA device (1 = A6000)")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "checkpoints" / "causal-toy"))
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
        from boldt_embed import train as T
    except ImportError as exc:
        raise SystemExit(f"Needs training extras: pip install -e '.[train]'. ({exc})")

    cfg = load_causal_config(args.config)
    triples = datamod.load_jsonl(TRIPLES)
    data = json.loads(BENCH.read_text(encoding="utf-8"))
    pooling = "mean" if cfg.pooling == "mean" else "eos"

    print("=== Evaluating BASE model (before training) ===")
    base_eval = evaluate_model(cfg.model_name_or_path, data, cfg, pooling, args.device_index)
    print("base ndcg@10:", base_eval["aggregate"]["ndcg@10"])

    print("=== REAL training on GPU ===")
    train_report = T.train_causal_real(
        cfg, triples, output_dir=args.out, device_index=args.device_index,
        epochs=args.epochs, lr=args.lr,
    )

    print("=== Evaluating TRAINED model ===")
    trained_eval = evaluate_model(args.out, data, cfg, pooling, args.device_index)
    print("trained ndcg@10:", trained_eval["aggregate"]["ndcg@10"])

    report = {
        "status": "ok",
        "scale_disclaimer": (
            "Tiny REAL run on the toy German triples to prove the GPU training + eval "
            "pipeline. NOT a production model and NOT a public-benchmark claim."
        ),
        "run_metadata": {
            "command": "scripts/run_real_training.py",
            "commit": _git_commit(),
            "date": "2026-05-29",
            "hardware": platform.platform(),
            "gpu": train_report.get("gpu_name"),
            "torch": __import__("torch").__version__,
        },
        "training": train_report,
        "eval_base_model": base_eval["aggregate"],
        "eval_trained_model": trained_eval["aggregate"],
        "eval_trained_per_query": trained_eval["queries"],
    }
    out_dir = ROOT / "outputs" / "real-training"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "real-training-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in
                      ("eval_base_model", "eval_trained_model")}, indent=2))
    print("saved:", out_dir / "real-training-report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
