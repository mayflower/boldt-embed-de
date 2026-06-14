#!/usr/bin/env python3
"""Build realistic fixed top-k RAG candidate lists from multiple first stages (stdlib, no ML).

Merges BM25 + v3 dense (+ optional e5 / qwen dense / WebFAQ hard negatives) top-k per query,
preserves each candidate's source, dedups by doc_id/text hash, labels (train: high-precision
teacher labels, uncertain=null; eval: labels null, positives from qrels), and writes a
candidate-list JSONL + a first-stage-recall report.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import rag_candidates as RC  # noqa: E402


def _read(path):
    p = pathlib.Path(path)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def _results_map(path):
    """per-source result file -> {query_id: ranked list}."""
    m = {}
    for r in _read(path):
        m[str(r.get("query_id"))] = r.get("candidates") or r.get("results") or r.get("doc_ids") or []
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--qrels", default=None)
    ap.add_argument("--bm25-results", default=None)
    ap.add_argument("--dense-results", default=None, help="v3 causal dense results")
    ap.add_argument("--e5-results", default=None)
    ap.add_argument("--qwen-results", default=None)
    ap.add_argument("--webfaq-hardnegs", default=None, help="{query_id, doc_ids:[...]} JSONL")
    ap.add_argument("--teacher-scores", default=None, help="{query_id, doc_id, reranker_score} JSONL")
    ap.add_argument("--mode", choices=["train", "eval"], default="train")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    queries = _read(args.queries)
    corpus = {str(d.get("doc_id") or d.get("id")): d for d in _read(args.corpus)}
    positives_by_query = {}
    for r in _read(args.qrels):
        if float(r.get("relevance", 1)) > 0:
            positives_by_query.setdefault(str(r["query_id"]), set()).add(str(r["doc_id"]))

    source_results = {}
    for name, path in (("bm25", args.bm25_results), ("v3_dense", args.dense_results),
                       ("e5_dense", args.e5_results), ("qwen_dense", args.qwen_results)):
        if path:
            source_results[name] = _results_map(path)
    hard_negs = {str(r["query_id"]): r.get("doc_ids", []) for r in _read(args.webfaq_hardnegs)} \
        if args.webfaq_hardnegs else None
    teacher_scores = {(str(r["query_id"]), str(r["doc_id"])): r.get("reranker_score")
                      for r in _read(args.teacher_scores)} if args.teacher_scores else None

    rows, report = RC.build_candidate_lists(
        queries, corpus, source_results, positives_by_query=positives_by_query,
        teacher_scores=teacher_scores, hard_negatives=hard_negs,
        top_k=args.top_k, is_eval=(args.mode == "eval"))
    print(f"[rag-candidates] {json.dumps(report, ensure_ascii=False)}")

    if args.dry_run:
        assert "torch" not in sys.modules, "dry-run must not import torch"
        print("dry-run-ok (no ML imports)")
        return 0

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(f"[write] {len(rows)} candidate-list rows -> {out}; report -> {out.with_suffix('.report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
