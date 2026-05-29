#!/usr/bin/env python3
"""Baseline eval on held-out GerDaLIR (legal) to contextualize our model's number.

Runs a strong off-the-shelf multilingual embedder (default intfloat/multilingual-e5-base)
under the SAME retrieval harness/metrics, so we know whether nDCG@10 ~0.027 means "our model
is weak" or "GerDaLIR is just hard". Requires sentence-transformers + a GPU.
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

from boldt_embed.metrics import aggregate, metrics_for_query  # noqa: E402

KS = (1, 10, 100)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load_gerdalir():
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
    queries = [{"id": str(q["_id"]), "text": q["text"], "positive_ids": qrels[str(q["_id"])]}
               for q in queries_ds if str(q["_id"]) in qrels]
    return corpus, queries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(f"Needs: pip install -e '.[eval]'. ({exc})")

    dev = f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu"
    is_e5 = "e5" in args.model.lower()
    qpref, dpref = ("query: ", "passage: ") if is_e5 else ("", "")
    model = SentenceTransformer(args.model, device=dev)

    corpus, queries = load_gerdalir()
    print(f"[baseline] model={args.model} corpus={len(corpus)} queries={len(queries)} e5_prefix={is_e5}")

    c_emb = model.encode([dpref + c["text"] for c in corpus], batch_size=args.batch_size,
                         normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    q_emb = model.encode([qpref + q["text"] for q in queries], batch_size=args.batch_size,
                         normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    corpus_ids = [c["id"] for c in corpus]
    sims = q_emb @ c_emb.t()
    _, topi = torch.topk(sims, min(max(KS), len(corpus_ids)), dim=1)
    rows = [metrics_for_query([corpus_ids[j] for j in topi[i].tolist()],
                              set(queries[i]["positive_ids"]), KS) for i in range(len(queries))]
    agg = aggregate(rows)

    report = {
        "status": "ok", "baseline_model": args.model, "benchmark": "mteb/GerDaLIRSmall (legal)",
        "corpus": len(corpus), "queries": len(queries), "aggregate": agg,
        "run_metadata": {"command": "scripts/eval_baseline.py", "commit": _git_commit(),
                         "date": "2026-05-29", "hardware": platform.platform(),
                         "gpu": torch.cuda.get_device_name(0) if dev.startswith("cuda") else "cpu"},
    }
    out = ROOT / "outputs" / "real-training" / "baseline-gerdalir-report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model": args.model, "gerdalir_aggregate": agg}, indent=2))
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
