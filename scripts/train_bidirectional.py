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
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Real training needs extras: pip install -e '.[train]' (and 'llm2vec', and a GPU)."
        ) from exc
    raise SystemExit(
        "Real MNTP + contrastive loop is intentionally not implemented in this scaffold. "
        "Wire LLM2Vec (bidirectional mask -> MNTP -> contrastive -> merge) here."
    )


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
