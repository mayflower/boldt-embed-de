#!/usr/bin/env python3
"""Build a reusable leakage blocking-index from one or more held-out eval corpora.

Stdlib, no ML, no network. Each eval row is expanded into text units over its text fields
(query/document/text/context/title); the dataset tag defaults to the file stem (override with
``name=path`` syntax). Writes a JSON index that `run_full_leakage_scan.py --index` can load.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import leakage_index as li  # noqa: E402


def _iter_eval_units(specs):
    """specs: list of 'path' or 'dataset=path'. Yields (eval_id, dataset, field, text)."""
    for spec in specs:
        dataset, _, path = spec.partition("=") if "=" in spec else ("", "", spec)
        path = path or spec
        dataset = dataset or pathlib.Path(path).stem
        p = pathlib.Path(path)
        if not p.exists():
            print(f"ERROR: eval corpus not found: {path}", file=sys.stderr)
            raise SystemExit(2)
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield from li.eval_texts_from_record(json.loads(line), dataset)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-corpus", nargs="+", required=True, help="eval JSONL(s) or dataset=path")
    ap.add_argument("--output", required=True, help="index JSON output path")
    ap.add_argument("--shingle-n", type=int, default=li.DEFAULT_SHINGLE_N)
    ap.add_argument("--num-perm", type=int, default=li.DEFAULT_NUM_PERM)
    args = ap.parse_args()

    units = list(_iter_eval_units(args.eval_corpus))
    index = li.build_eval_leakage_index(units, shingle_n=args.shingle_n, num_perm=args.num_perm)
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index.to_dict(), ensure_ascii=False), encoding="utf-8")
    print(f"[index] {index.n_eval_texts()} eval text units from {len(args.eval_corpus)} corpus "
          f"file(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
