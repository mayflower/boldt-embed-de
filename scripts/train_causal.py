#!/usr/bin/env python3
"""Causal embedder training entrypoint.

``--dry-run`` validates the config and instruction wiring on the toy German triples
without loading any weights (pure stdlib). A real run requires ``pip install -e '.[train]'``,
the base weights, and a GPU.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data  # noqa: E402
from boldt_embed.config import load_causal_config  # noqa: E402
from boldt_embed.model_causal import CausalEmbedder  # noqa: E402

TOY_TRIPLES = ROOT / "data" / "samples" / "toy_triples_de.jsonl"


def dry_run(config_path: str) -> dict:
    cfg = load_causal_config(config_path)
    embedder = CausalEmbedder(cfg)
    triples = data.load_jsonl(TOY_TRIPLES)
    report = data.validate_dataset(triples)
    if not report.ok:
        return {"status": "fail", "errors": report.errors}
    queries = [t["query"] for t in triples]
    positives = [t["positive"] for t in triples]
    out = embedder.dry_run(queries, positives)
    out["data_records"] = report.num_records
    out["data_with_negatives"] = report.num_with_negatives
    out["config"] = str(config_path)
    return out


def real_train(config_path: str) -> dict:  # pragma: no cover - requires extras + GPU
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Real training needs the training extras: pip install -e '.[train]' (and a GPU)."
        ) from exc
    raise SystemExit(
        "Real training loop is intentionally not implemented in this scaffold. "
        "Wire SentenceTransformers MNRL + Matryoshka here once weights and licensed data exist."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "training_causal.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    args = parser.parse_args()

    if not args.dry_run:
        real_train(args.config)
        return 0

    report = dry_run(args.config)
    if args.format == "markdown":
        lines = [f"# Causal Embedder Dry-Run", "", f"Status: **{report.get('status')}**", ""]
        for k, v in report.items():
            if k != "status":
                lines.append(f"- {k}: {v}")
        print("\n".join(lines))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
