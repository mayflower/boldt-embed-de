#!/usr/bin/env python3
"""Build RAG eval sets (stdlib, no ML, no network).

Two modes:
  webfaq   deterministic held-out FAQ eval from real FAQ rows (hash split; train/dev/test never
           share a (query, answer) pair). Writes corpus/queries/qrels for the chosen --split.
  local    assemble the local RAG eval already dropped at data/eval/rag_local/{corpus,queries,
           qrels}.jsonl into a validated eval set (copies through, validated).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import rag_eval_schema as R  # noqa: E402


def _read(path):
    return [json.loads(l) for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _write(path, rows):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):6d} -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["webfaq", "local"], default="webfaq")
    ap.add_argument("--faq-input", default=None, help="real FAQ rows JSONL (webfaq mode)")
    ap.add_argument("--split", choices=["dev", "test", "train"], default="test")
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--local-dir", default=str(ROOT / "data" / "eval" / "rag_local"))
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.output_dir)
    if args.mode == "webfaq":
        if not args.faq_input or not pathlib.Path(args.faq_input).exists():
            print(f"ERROR: --faq-input required and must exist", file=sys.stderr)
            return 2
        rows = _read(args.faq_input)
        corpus, queries, qrels = R.build_webfaq_eval(rows, split=args.split,
                                                     dev_frac=args.dev_frac, test_frac=args.test_frac)
    else:  # local
        ld = pathlib.Path(args.local_dir)
        corpus = _read(ld / "corpus.jsonl")
        queries = _read(ld / "queries.jsonl")
        qrels = _read(ld / "qrels.jsonl") if (ld / "qrels.jsonl").exists() else []

    errors = R.validate_eval_set(queries, corpus, qrels or None)
    if errors:
        print(f"ERROR: built eval set is invalid:\n  " + "\n  ".join(errors[:10]), file=sys.stderr)
        return 1
    _write(out / "corpus.jsonl", corpus)
    _write(out / "queries.jsonl", queries)
    _write(out / "qrels.jsonl", qrels)
    print(f"[rag-eval] mode={args.mode} split={args.split}: {len(queries)} queries, "
          f"{len(corpus)} docs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
