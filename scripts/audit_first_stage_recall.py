#!/usr/bin/env python3
"""Audit first-stage retrieval recall for an eval set BEFORE training another reranker (stdlib, no
ML). Reports recall@k, positive_in_top_k, oracle nDCG@10 (realistic retriever-only vs perfect
candidate set), the candidate-source contribution (BM25 / dense / overlap / union / injected-only),
missing-positive breakdown, the realistic upper-bound reranker lift, and deterministic examples — so
we know whether the bottleneck is dense retrieval, BM25 retrieval, candidate-list construction, or
reranker quality.

Candidate-lists input: a JSONL where each row is {query_id, positive_doc_ids, candidates:[{doc_id,
candidate_source, first_stage_rank, first_stage_score, ...}]}. Optional --queries/--qrels/--corpus
override/augment positives + example text. Injected/oracle sources (manual/gold/…) are NOT counted
as retriever hits.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import first_stage_audit as FA  # noqa: E402


def _read_jsonl(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]


def _load_qrels(p):
    qrels = {}
    for r in _read_jsonl(p):
        qid = str(r.get("query_id"))
        pos = set(r.get("positive_doc_ids") or [])
        if r.get("positive_doc_id"):
            pos.add(r["positive_doc_id"])
        if r.get("doc_id") and (r.get("relevance", 1) or r.get("label", 1)):
            pos.add(r["doc_id"])
        qrels.setdefault(qid, set()).update(p for p in pos if p)
    return qrels


def _load_map(p, key, val):
    out = {}
    for r in _read_jsonl(p):
        if r.get(key) is not None:
            out[str(r[key])] = r.get(val, "")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--candidate-lists", required=True)
    ap.add_argument("--queries", default=None)
    ap.add_argument("--qrels", default=None)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--ks", default="10,20,50,100,200")
    ap.add_argument("--max-examples", type=int, default=8)
    args = ap.parse_args()

    rows = _read_jsonl(args.candidate_lists)
    qrels = _load_qrels(args.qrels) if args.qrels else None
    corpus = _load_map(args.corpus, "doc_id", "text") if args.corpus else None
    queries = _load_map(args.queries, "query_id", "query") if args.queries else None
    if corpus is None:  # fall back to candidate text for examples
        corpus = {c["doc_id"]: c.get("text", "") for r in rows for c in (r.get("candidates") or [])
                  if c.get("doc_id")}
    if queries is None:
        queries = {str(r["query_id"]): r.get("query", "") for r in rows if r.get("query_id")}
    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())

    report = FA.audit_set(rows, name=args.eval_set, ks=ks, qrels=qrels, corpus=corpus,
                          queries=queries, max_examples=args.max_examples)
    assert "torch" not in sys.modules, "audit must not import torch"

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    if args.markdown:
        pathlib.Path(args.markdown).write_text(_render_md(report), encoding="utf-8")
    b = report["bottleneck"]
    csc = report["candidate_source_contribution"]
    print(f"[recall-audit] {args.eval_set}: recall@10 {report['recall']['recall@10']} "
          f"missing {report['missing_positive_rate']} bottleneck={b['primary']} "
          f"dense_source={csc['has_dense_source']} ceiling={b['realistic_reranker_ceiling']} "
          f"-> {args.output}")
    return 0


def _render_md(r) -> str:
    csc = r["candidate_source_contribution"]
    b = r["bottleneck"]
    L = [f"# First-stage recall audit — {r['eval_set']}", "",
         f"**Bottleneck: `{b['primary']}`.** {b['detail']}", "",
         f"_{r['n_queries']} queries, {r['total_positives']} positives. A reranker can only reorder "
         "what the first stage retrieved — injected/oracle sources (e.g. `manual`) are NOT counted as "
         "retriever hits._", "",
         "## Recall & nDCG", "",
         "| metric | value |", "|---|--:|"]
    for k, v in r["recall"].items():
        L.append(f"| {k} | {v} |")
    L += [f"| positive_in_top_10 rate | {r['positive_in_top_10_rate']} |",
          f"| first-stage nDCG@10 | {r['first_stage_ndcg10']} |",
          f"| oracle nDCG@10 (retriever-only, realistic) | {r['oracle_ndcg10_retriever']} |",
          f"| oracle nDCG@10 (with injected positives) | {r['oracle_ndcg10_with_injected']} |",
          f"| **upper-bound reranker lift (realistic)** | **{r['upper_bound_reranker_lift_realistic']}** |",
          f"| upper-bound lift IF candidate set were perfect | {r['upper_bound_reranker_lift_if_perfect_candidates']} |",
          f"| illusory lift from injected positives | {r['illusory_lift_from_injection']} |", "",
          "## Missing positives (reranker can NEVER recover these)", "",
          f"- missing_positive_count: **{r['missing_positive_count']}** "
          f"(rate **{r['missing_positive_rate']}**)",
          f"- by domain: {r['missing_by_domain']}",
          f"- by query_style: {r['missing_by_query_style']}",
          f"- by source-of-missing: {r['missing_by_source']}", "",
          "## Candidate-source contribution", "",
          "| bucket | positives |", "|---|--:|",
          f"| BM25 hits | {csc['bm25_hits']} |",
          f"| dense hits | {csc['dense_hits']} |",
          f"| other-retriever hits | {csc['other_retriever_hits']} |",
          f"| BM25-only | {csc['bm25_only']} |",
          f"| dense-only | {csc['dense_only']} |",
          f"| overlap (BM25 ∧ dense) | {csc['overlap_bm25_and_dense']} |",
          f"| union (any retriever) | {csc['union_any_retriever']} |",
          f"| injected-only (NOT retrieved) | {csc['injected_only_not_retrieved']} |",
          f"| absent from list | {csc['absent_from_list']} |", "",
          f"**Has a dense first stage in these lists: {csc['has_dense_source']}.** "
          + ("No dense candidate source is present — these lists are BM25-only (plus injected gold), "
             "so dense-vs-BM25 recall cannot be compared from this data." if not csc['has_dense_source']
             else ""), "",
          "## Examples (deterministic)", ""]
    for ex in r["examples"]:
        L.append(f"- `{ex['query_id']}` [{ex['query_style']}] — {ex['note']}; "
                 f"first-stage nDCG@10 {ex['first_stage_ndcg10']}, "
                 f"retriever-oracle {ex['oracle_ndcg10_retriever']}. "
                 f"missing positive `{ex['missing_positive']}`"
                 + (f": “{ex['query']}”" if ex['query'] else ""))
    if not r["examples"]:
        L.append("- (none — every positive was retrieved by a real first stage)")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
