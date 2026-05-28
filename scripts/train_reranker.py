#!/usr/bin/env python3
"""Reranker training entrypoint.

``--dry-run`` validates the cross-encoder input template on toy relevant/irrelevant pairs
without weights. A real run requires ``pip install -e '.[train]'`` and a GPU.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data  # noqa: E402
from boldt_embed.config import load_reranker_config  # noqa: E402
from boldt_embed.reranker import Reranker  # noqa: E402

TOY_TRIPLES = ROOT / "data" / "samples" / "toy_triples_de.jsonl"


def dry_run(config_path: str) -> dict:
    cfg = load_reranker_config(config_path)
    reranker = Reranker(cfg)
    triples = data.load_jsonl(TOY_TRIPLES)
    # Each triple yields one relevant pair (query, positive) and one or more irrelevant
    # pairs (query, negative) — the cross-encoder's training signal.
    pairs = []
    for t in triples:
        pairs.append({"query": t["query"], "document": t["positive"], "label": cfg.positive_label})
        for neg in t.get("negatives", []):
            pairs.append({"query": t["query"], "document": neg, "label": cfg.negative_label})
    sample = triples[0]
    docs = [sample["positive"]] + sample.get("negatives", [])
    out = reranker.dry_run(sample["query"], docs)
    out["training_pairs"] = len(pairs)
    out["positive_pairs"] = sum(1 for p in pairs if p["label"] == cfg.positive_label)
    out["negative_pairs"] = sum(1 for p in pairs if p["label"] == cfg.negative_label)
    out["config"] = str(config_path)
    return out


def real_train(config_path: str) -> dict:  # pragma: no cover - requires extras + GPU
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Real training needs: pip install -e '.[train]' (and a GPU).") from exc
    raise SystemExit(
        "Real cross-encoder training loop is intentionally not implemented in this scaffold."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "training_reranker.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    args = parser.parse_args()

    if not args.dry_run:
        real_train(args.config)
        return 0

    report = dry_run(args.config)
    if args.format == "markdown":
        lines = ["# Reranker Dry-Run", "", f"Status: **{report.get('status')}**", ""]
        for k, v in report.items():
            if k != "status":
                lines.append(f"- {k}: {v}")
        print("\n".join(lines))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
