#!/usr/bin/env python3
"""Prepare the bidirectional Boldt student (LLM2Vec-style): enable + verify bidirectional
attention, then run MNTP pre-adaptation and export a bi-encoder (Prompt 5).

`--dry-run` validates the plan and counts the MNTP texts WITHOUT importing torch. The real
path enables bidirectional attention, **verifies** it numerically, runs MNTP, and exports —
and fails with a clear message if extras/GPU are missing.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data_pipeline as dp  # noqa: E402  (stdlib)
from boldt_embed.config import load_bidirectional_config  # noqa: E402


def _read_texts(path):
    texts = []
    if path and pathlib.Path(path).exists():
        for row in dp.stream_jsonl(path):
            t = row.get("text") or row.get("document") or ""
            if t.strip():
                texts.append(t)
    return texts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "training_bidirectional.json"))
    ap.add_argument("--base-model", default=None, help="override config model_name_or_path")
    ap.add_argument("--texts", default=str(ROOT / "data" / "processed" / "mntp_texts.jsonl"))
    ap.add_argument("--output", default=str(ROOT / "outputs" / "checkpoints" / "boldt-bi-mntp"))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_bidirectional_config(args.config)
    base_model = args.base_model or cfg.model_name_or_path
    texts = _read_texts(args.texts)
    plan = {"base_model": base_model, "adaptation": cfg.adaptation,
            "pooling": args.pooling, "pooling_ablation": cfg.pooling_ablation,
            "steps": args.steps, "batch_size": args.batch_size, "max_length": args.max_length,
            "num_mntp_texts": len(texts), "output": args.output}
    print(f"[plan] {json.dumps(plan, ensure_ascii=False)}")

    if args.dry_run:
        if not texts:
            print(f"[dry-run] note: no MNTP texts at {args.texts} (provide before a real run)")
        assert "torch" not in sys.modules, "dry-run must not import torch"
        print("dry-run-ok (no ML imports)")
        return 0

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"Real preparation needs extras: pip install -e '.[train]'. ({exc})")
    if not texts:
        print(f"ERROR: no MNTP texts found at {args.texts}", file=sys.stderr)
        return 2

    from boldt_embed import llm2vec_boldt as B
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model, tok = B.load_boldt_for_bidirectional(base_model, device=device,
                                                dtype="bfloat16" if args.bf16 else "float32")
    verdict = B.verify_bidirectional_attention(model, tok, device=device)
    print(f"[verify] {json.dumps(verdict, ensure_ascii=False)}")
    if not verdict["is_bidirectional"]:
        print("ERROR: attention is not bidirectional after patching; aborting.", file=sys.stderr)
        return 3
    stats = B.run_mntp_adaptation(model, tok, texts, steps=args.steps,
                                  batch_size=args.batch_size, max_length=args.max_length,
                                  device=device)
    print(f"[mntp] {json.dumps(stats, ensure_ascii=False)}")
    out = B.export_bi_encoder(model, tok, args.output, pooling=args.pooling)
    print(f"[export] bi-encoder -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
