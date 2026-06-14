#!/usr/bin/env python3
"""Build v3 reranker candidate lists: high-precision labels + source-balanced lists (stdlib).

Fixes v2's residual GermanQuAD degradation by (a) using only HIGH-PRECISION positives (stricter
teacher reranker threshold, from calibration), (b) labeling clear negatives only, leaving
uncertain candidates as ``label=null`` (listwise soft targets, never hard BCE negatives), and
(c) preserving candidates from multiple first stages (BM25 / student-dense / e5-or-teacher-dense
/ teacher-reranker-mined) so the reranker does not overfit one candidate distribution.

The teacher cache (calibrated `qwen3_v3.filtered_reranker.jsonl` or the full cache) supplies the
candidate documents + teacher reranker scores; the per-source result files supply which source
surfaced each (query, doc) — used to tag ``candidate_source`` and enforce source diversity.
NEVER reads eval corpora: candidates come only from training-side sources.

No ML, no network.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data_pipeline as dp  # noqa: E402
from boldt_embed import reranker_modern as RM  # noqa: E402
from boldt_embed import teacher as T  # noqa: E402


def _doc_id_map(path):
    """A per-source result file -> {query_id: set(doc_id)}. Rows: {query_id, candidates|results|
    doc_ids:[doc_id | {doc_id}]}."""
    m = {}
    if not path or not pathlib.Path(path).exists():
        return m
    for r in dp.stream_jsonl(path):
        qid = str(r.get("query_id"))
        ids = r.get("candidates") or r.get("results") or r.get("doc_ids") or []
        s = m.setdefault(qid, set())
        for it in ids:
            s.add(str(it.get("doc_id") if isinstance(it, dict) else it))
    return m


def build(cache_rows, source_maps, *, positive_threshold=RM.V3_POSITIVE_THRESHOLD,
          neg_margin=RM.V3_NEG_MARGIN, min_sources=3):
    """source_maps: {source_name: {query_id: set(doc_id)}}."""
    by_q = {}
    for r in cache_rows:
        by_q.setdefault(str(r["query_id"]), []).append(r)

    rows, skipped_no_pos = [], 0
    for qid, grp in by_q.items():
        # gold positives = positive rows clearing the high-precision threshold
        gold = [r for r in grp if r.get("positive") is True
                and r.get("reranker_score") is not None
                and float(r["reranker_score"]) >= positive_threshold]
        if not gold:
            skipped_no_pos += 1
            continue
        gold_ids = {str(r["doc_id"]) for r in gold}
        query = grp[0].get("query", "")
        cands = []
        for r in grp:
            did = str(r["doc_id"])
            ts = r.get("reranker_score")
            label = 1 if did in gold_ids else RM.v3_label(ts, positive_threshold, neg_margin)
            srcs = sorted(name for name, mp in source_maps.items() if did in mp.get(qid, set()))
            cands.append({
                "doc_id": did, "document": r.get("document", ""), "label": label,
                "teacher_score": float(ts) if ts is not None else None,
                "candidate_source": "+".join(srcs) if srcs else "teacher_cache",
                "domain": r.get("domain", "unknown"), "synthetic": bool(r.get("synthetic"))})
        rows.append({"query_id": qid, "query": query, "candidates": cands,
                     "positive_doc_ids": sorted(gold_ids),
                     "domain": grp[0].get("domain", "unknown"),
                     "source": grp[0].get("source", "unknown")})

    summary = RM.reranker_training_summary(rows, positive_threshold)
    # source diversity: distinct candidate_source tags per list
    div = []
    for r in rows:
        tags = set()
        for c in r["candidates"]:
            tags.update((c.get("candidate_source") or "").split("+"))
        div.append(len([t for t in tags if t]))
    summary["queries"] = len(rows)
    summary["skipped_no_high_precision_positive"] = skipped_no_pos
    summary["lists_with_min_sources"] = sum(1 for d in div if d >= min_sources)
    summary["avg_sources_per_list"] = round(sum(div) / len(div), 2) if div else 0.0
    summary["min_sources_required"] = min_sources
    return rows, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-cache", required=True,
                    help="calibrated reranker positives / teacher cache JSONL")
    ap.add_argument("--bm25-results", default=None)
    ap.add_argument("--dense-results", default=None, help="student dense (causal v2/v3) results")
    ap.add_argument("--e5-results", default=None)
    ap.add_argument("--teacher-reranker-results", default=None, help="teacher reranker-mined doc lists")
    ap.add_argument("--output", required=True)
    ap.add_argument("--positive-threshold", type=float, default=RM.V3_POSITIVE_THRESHOLD)
    ap.add_argument("--neg-margin", type=float, default=RM.V3_NEG_MARGIN)
    ap.add_argument("--min-sources", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not pathlib.Path(args.teacher_cache).exists():
        print(f"ERROR: teacher cache not found: {args.teacher_cache}", file=sys.stderr)
        return 2
    cache_rows = T.read_teacher_cache_jsonl(args.teacher_cache)
    source_maps = {}
    for name, path in (("bm25", args.bm25_results), ("student_dense", args.dense_results),
                       ("e5_dense", args.e5_results),
                       ("teacher_reranker", args.teacher_reranker_results)):
        if path:
            source_maps[name] = _doc_id_map(path)

    rows, summary = build(cache_rows, source_maps, positive_threshold=args.positive_threshold,
                          neg_margin=args.neg_margin, min_sources=args.min_sources)
    summary["candidate_source_files"] = sorted(source_maps.keys())
    print(f"[v3-reranker-candidates] {json.dumps(summary, ensure_ascii=False)}")

    if args.dry_run:
        assert "torch" not in sys.modules, "dry-run must not import torch"
        print("dry-run-ok (no ML imports)")
        return 0

    n = dp.write_jsonl(args.output, rows)
    rep = pathlib.Path(args.output).with_suffix(".summary.json")
    rep.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {n} candidate-list rows -> {args.output}; summary -> {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
