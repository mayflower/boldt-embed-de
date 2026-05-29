#!/usr/bin/env python3
"""Export a trained causal/bi checkpoint to a SentenceTransformers-compatible model.

Requires: pip install -e '.[eval]'  (sentence-transformers). The exported folder contains
modules.json + a Transformer + a Pooling module and loads with `SentenceTransformer(path)`,
which is what the MTEB harness expects.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def export(checkpoint: str, out_dir: str, pooling: str = "eos", max_seq_length: int = 256) -> str:
    from sentence_transformers import SentenceTransformer, models

    mode = "lasttoken" if pooling in ("eos", "last_token", "eos_or_last_token") else "mean"
    word = models.Transformer(checkpoint, max_seq_length=max_seq_length)
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode=mode)
    st = SentenceTransformer(modules=[word, pool])
    st.save(out_dir)
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="trained model dir (save_pretrained)")
    parser.add_argument("--out", required=True, help="output SentenceTransformers dir")
    parser.add_argument("--pooling", default="eos", choices=["eos", "mean", "last_token"])
    parser.add_argument("--max-seq-length", type=int, default=256)
    args = parser.parse_args()

    try:
        import sentence_transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"Needs eval extras: pip install -e '.[eval]'. ({exc})")

    out = export(args.checkpoint, args.out, args.pooling, args.max_seq_length)
    # smoke-encode to confirm the exported model loads and runs
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(out)
    vec = st.encode(["Hallo Welt, dies ist ein Test."], normalize_embeddings=True)
    print(f"exported to {out}; dim={len(vec[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
