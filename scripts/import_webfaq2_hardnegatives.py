#!/usr/bin/env python3
"""Import WebFAQ / WebFAQ 2.0 hard negatives as a first-class v5 training source (no network by
default). Converts to BOTH embedder triplets (with teacher margins) and reranker candidate lists
(with listwise teacher scores). Fails closed on a missing local file, unknown/absent license, or a
record without a positive cross-encoder score. The Hugging Face path is opt-in via --download-hf.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import webfaq2_loader as W  # noqa: E402


def _write_jsonl(path: pathlib.Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", help="local WebFAQ2 hard-negative JSONL (default path)")
    ap.add_argument("--output", required=True, help="embedder-triplet JSONL output")
    ap.add_argument("--reranker-output", default=None,
                    help="reranker candidate-list JSONL (default: <output>.reranker_lists.jsonl)")
    ap.add_argument("--language", default="de")
    ap.add_argument("--min-cross-encoder-margin", type=float, default=W.DEFAULT_MIN_MARGIN)
    ap.add_argument("--max-negatives-per-query", type=int, default=W.DEFAULT_MAX_NEGATIVES)
    ap.add_argument("--false-negative-margin", type=float, default=W.DEFAULT_FALSE_NEGATIVE_MARGIN)
    ap.add_argument("--report", default="outputs/v5-small-rag/webfaq2_hardneg_report.json")
    ap.add_argument("--download-hf", action="store_true",
                    help="opt-in: fetch from Hugging Face (lazy `datasets` import; network)")
    ap.add_argument("--hf-dataset", default=None,
                    help="HF dataset id (REQUIRED with --download-hf; confirm against WebFAQ2 release)")
    ap.add_argument("--hf-limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.download_hf:
        if not args.hf_dataset:
            print("ERROR: --download-hf requires --hf-dataset (no default; confirm the id)",
                  file=sys.stderr)
            return 2
        records = W.load_from_hf(args.hf_dataset, args.language, limit=args.hf_limit)
    else:
        if not args.input or not pathlib.Path(args.input).exists():
            print(f"ERROR: local --input not found (fail closed; use --download-hf to fetch): "
                  f"{args.input}", file=sys.stderr)
            return 2
        records = W.load_local_jsonl(args.input)

    if not records:
        print("ERROR: no WebFAQ2 records read (fail closed)", file=sys.stderr)
        return 1

    out = W.import_webfaq2(
        records, language=args.language, min_margin=args.min_cross_encoder_margin,
        max_negatives=args.max_negatives_per_query,
        false_negative_margin=args.false_negative_margin)
    report = out["report"]

    pathlib.Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    if not args.download_hf:
        assert "datasets" not in sys.modules, "local/dry-run path must not touch the network"
    assert "torch" not in sys.modules, "import must not load torch"

    npq = report["negatives_per_query"]
    print(f"[webfaq2] status={report['status']} queries={report['imported_queries']} "
          f"triplets={report['embedder_triplets']} lists={report['reranker_lists']} "
          f"neg/query(avg)={npq['avg']} dropped_false_neg={report['dropped_false_negatives']} "
          f"-> {args.report}")
    for e in report["errors"][:5]:
        print(f"  ✗ {e}", file=sys.stderr)

    if report["status"] != "pass":
        print("FAIL — WebFAQ2 import rejected (fail closed; see report)", file=sys.stderr)
        return 1

    if args.dry_run:
        print("dry-run-ok (no network/ML; report written, data files NOT written)")
        return 0

    rer_out = pathlib.Path(args.reranker_output) if args.reranker_output \
        else pathlib.Path(args.output).with_suffix(".reranker_lists.jsonl")
    _write_jsonl(pathlib.Path(args.output), out["triplets"])
    _write_jsonl(rer_out, out["reranker_lists"])
    print(f"[webfaq2] wrote {len(out['triplets'])} triplets -> {args.output}; "
          f"{len(out['reranker_lists'])} reranker lists -> {rer_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
