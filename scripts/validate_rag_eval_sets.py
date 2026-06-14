#!/usr/bin/env python3
"""Validate a RAG eval set (stdlib): schema + every positive_doc_id in corpus, and optionally
that no eval pair leaks into a training candidate file. Exit non-zero on any problem.
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
    p = pathlib.Path(path)
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--qrels", default=None)
    ap.add_argument("--candidate-lists", default=None,
                    help="optional fixed candidate-list JSONL — each must contain a positive")
    ap.add_argument("--train-candidates", default=None,
                    help="optional training candidate file — must NOT contain any eval pair")
    args = ap.parse_args()

    queries = _read(args.queries) or []
    corpus = _read(args.corpus) or []
    qrels = _read(args.qrels) if args.qrels else None

    errors = R.validate_eval_set(queries, corpus, qrels)

    if args.candidate_lists:
        for i, cl in enumerate(_read(args.candidate_lists) or []):
            errors += [f"candidate_lists[{i}]: {x}"
                       for x in R.validate_candidate_list(cl, require_positive=True)]

    if args.train_candidates:
        eval_qids = {q["query_id"] for q in queries if isinstance(q.get("query_id"), str)}
        eval_dids = {d["doc_id"] for d in corpus if isinstance(d.get("doc_id"), str)}
        errors += R.check_no_eval_leakage(_read(args.train_candidates) or [], eval_qids, eval_dids)

    report = {"status": "ok" if not errors else "fail", "n_queries": len(queries),
              "n_corpus": len(corpus), "n_errors": len(errors), "errors": errors[:50]}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        print(f"FAIL — {len(errors)} problem(s) in the RAG eval set", file=sys.stderr)
        return 1
    print(f"OK — {len(queries)} queries / {len(corpus)} docs valid; all positives in corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
