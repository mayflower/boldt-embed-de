#!/usr/bin/env python3
"""REAL reranker training on GPU + a reranking evaluation report.

Trains a LlamaForSequenceClassification cross-encoder on positive/hard-negative pairs, then
reranks an (id-order) shortlist of the toy benchmark and reports nDCG@10 before vs after.
Honest scale: tiny real run; the reranker is trained on 7 triples.
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

from boldt_embed import data as datamod  # noqa: E402
from boldt_embed.config import load_reranker_config  # noqa: E402
from boldt_embed.metrics import aggregate, metrics_for_query  # noqa: E402

TRIPLES = ROOT / "data" / "samples" / "toy_triples_de.jsonl"
BENCH = ROOT / "benchmarks" / "toy_de_retrieval.json"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def rerank_eval(model_path, cfg, data, device_index) -> dict:
    from boldt_embed import train as T

    corpus = data["corpus"]
    doc_ids = [d["id"] for d in corpus]
    doc_texts = [d["text"] for d in corpus]
    base_rows, rr_rows = [], []
    for q in data["queries"]:
        pos = set(q["positive_doc_ids"])
        base_rows.append(metrics_for_query(doc_ids, pos, (10,)))  # id-order baseline
        scores = T.rerank_scores_real(model_path, q["query"], doc_texts, cfg.input_template,
                                      device_index=device_index)
        ranked = [doc_ids[i] for i in sorted(range(len(doc_ids)), key=lambda j: scores[j], reverse=True)]
        rr_rows.append(metrics_for_query(ranked, pos, (10,)))
    return {"baseline_idorder": aggregate(base_rows), "reranked": aggregate(rr_rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "training_reranker.json"))
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "checkpoints" / "reranker-toy"))
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
        from boldt_embed import train as T
    except ImportError as exc:
        raise SystemExit(f"Needs training extras: pip install -e '.[train]'. ({exc})")

    cfg = load_reranker_config(args.config)
    triples = datamod.load_jsonl(TRIPLES)
    data = json.loads(BENCH.read_text(encoding="utf-8"))

    train_report = T.train_reranker_real(cfg, triples, output_dir=args.out,
                                         device_index=args.device_index, epochs=args.epochs)
    eval_report = rerank_eval(args.out, cfg, data, args.device_index)

    report = {
        "status": "ok",
        "scale_disclaimer": "Tiny REAL reranker run (7 triples). Proves training + reranking; not production.",
        "run_metadata": {
            "command": "scripts/run_real_reranker.py", "commit": _git_commit(),
            "date": "2026-05-29", "hardware": platform.platform(),
            "gpu": train_report.get("gpu_name"), "torch": __import__("torch").__version__,
        },
        "training": train_report,
        "reranking_eval": eval_report,
    }
    out_dir = ROOT / "outputs" / "real-training"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reranker-eval-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"train_pairwise_accuracy": train_report["train_pairwise_accuracy"],
                      "reranking_eval": eval_report}, indent=2))
    print("saved:", out_dir / "reranker-eval-report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
