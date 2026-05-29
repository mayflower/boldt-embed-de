#!/usr/bin/env python3
"""REAL bidirectional (LLM2Vec) training on GPU: bidirectional attention -> MNTP -> contrastive.

Requires extras + GPU. Honest scale: a tiny real run that proves the LLM2Vec pipeline
(verified bidirectional info-flow, MNTP denoising, contrastive), not a production model.
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
from boldt_embed.config import load_bidirectional_config  # noqa: E402

TRIPLES = ROOT / "data" / "samples" / "toy_triples_de.jsonl"
BENCH = ROOT / "benchmarks" / "toy_de_retrieval.json"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "training_bidirectional.json"))
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--mntp-steps", type=int, default=10)
    parser.add_argument("--contrastive-steps", type=int, default=12)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "checkpoints" / "bi-toy"))
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
        from boldt_embed import train as T
    except ImportError as exc:
        raise SystemExit(f"Needs training extras: pip install -e '.[train]'. ({exc})")

    cfg = load_bidirectional_config(args.config)
    triples = datamod.load_jsonl(TRIPLES)
    corpus = json.loads(BENCH.read_text(encoding="utf-8"))["corpus"]
    mntp_texts = [t["positive"] for t in triples] + [d["text"] for d in corpus]

    report = T.train_bidirectional_real(
        cfg, triples, mntp_texts, output_dir=args.out, device_index=args.device_index,
        mntp_steps=args.mntp_steps, contrastive_steps=args.contrastive_steps,
    )
    report["scale_disclaimer"] = (
        "Tiny REAL LLM2Vec run (bidirectional attention verified, MNTP + contrastive) to "
        "prove the pipeline. NOT a production model."
    )
    report["run_metadata"] = {
        "command": "scripts/run_real_bidirectional.py", "commit": _git_commit(),
        "date": "2026-05-29", "hardware": platform.platform(),
        "gpu": report.get("gpu_name"), "torch": __import__("torch").__version__,
    }
    out_dir = ROOT / "outputs" / "real-training"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bidirectional-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("mntp_initial_loss", "mntp_final_loss",
                       "contrastive_initial_loss", "contrastive_final_loss")}, indent=2))
    print("saved:", out_dir / "bidirectional-report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
