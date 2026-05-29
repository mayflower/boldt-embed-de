#!/usr/bin/env python3
"""Bidirectional (LLM2Vec/MNTP) embedder training entrypoint.

``--dry-run`` validates config + MNTP/merge plan on the toy German data without weights.
A real run requires ``pip install -e '.[train]'`` (and ideally the ``llm2vec`` package).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data  # noqa: E402
from boldt_embed.config import load_bidirectional_config  # noqa: E402
from boldt_embed.model_bidirectional import BidirectionalEmbedder  # noqa: E402

TOY_TRIPLES = ROOT / "data" / "samples" / "toy_triples_de.jsonl"


def dry_run(config_path: str) -> dict:
    cfg = load_bidirectional_config(config_path)
    embedder = BidirectionalEmbedder(cfg)
    triples = data.load_jsonl(TOY_TRIPLES)
    texts = [t["positive"] for t in triples]
    out = embedder.dry_run(texts)
    out["config"] = str(config_path)
    return out


def real_train(config_path: str) -> dict:  # pragma: no cover - requires extras + GPU
    """Real LLM2Vec training (train.train_bidirectional_real: bidirectional mask -> MNTP ->
    contrastive) on the toy data. For an at-scale run, point mntp_texts/triples at a real
    corpus (e.g. DT-de-dpr) — see scripts/run_real_bidirectional.py."""
    try:
        import torch  # noqa: F401
        from boldt_embed import train as T
    except ImportError as exc:
        raise SystemExit("Real training needs: pip install -e '.[train]' (and a GPU).") from exc
    cfg = load_bidirectional_config(config_path)
    triples = data.load_jsonl(TOY_TRIPLES)
    texts = [t["positive"] for t in triples]
    print("[note] training on TOY data; for scale use scripts/run_real_bidirectional.py with a real corpus")
    report = T.train_bidirectional_real(
        cfg, triples, texts, output_dir=str(ROOT / "outputs" / "checkpoints" / "bi-real"),
        mntp_steps=10, contrastive_steps=12)
    print(json.dumps({k: report[k] for k in ("mntp_final_loss", "contrastive_final_loss", "checkpoint")}, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "training_bidirectional.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    args = parser.parse_args()

    if not args.dry_run:
        real_train(args.config)
        return 0

    report = dry_run(args.config)
    if args.format == "markdown":
        lines = ["# Bidirectional Embedder Dry-Run", "", f"Status: **{report.get('status')}**", ""]
        for k, v in report.items():
            if k != "status":
                lines.append(f"- {k}: {v}")
        print("\n".join(lines))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
