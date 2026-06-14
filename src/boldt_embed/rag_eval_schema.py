"""RAG-reranker evaluation schemas, leakage-safe splits, and metrics (pure stdlib).

A RAG reranker is judged on whether it promotes answer-supporting passages **inside a fixed
top-k candidate set** — not on dense retrieval over a whole corpus. This module defines the
query / corpus / fixed-candidate-list schemas, a deterministic (hash-based) train/dev/test split
that never leaks eval pairs into training, validators, and the metrics:
nDCG@10, MRR@10, Recall@10, positive_in_top_10, answer_support_at_10, and reranker_delta_ndcg10.

No ML, no network.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .metrics import aggregate, metrics_for_query

CANDIDATE_SOURCES = frozenset({"bm25", "dense", "e5", "qwen", "manual", "webfaq_hardneg"})
ANSWER_TYPES = frozenset({"fact", "definition", "procedure", "faq", "multi_hop", "other"})
K = 10


# ------------------------------------------------------------------ stable hashing / split
def stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(str(s).encode("utf-8"), digest_size=8).digest(), "big")


def assign_split(key: str, dev_frac: float = 0.1, test_frac: float = 0.1) -> str:
    """Deterministic train/dev/test assignment from a stable hash of ``key`` (e.g. query_id).
    Same key -> same split across runs/machines (blake2b, not Python's salted hash)."""
    bucket = (stable_hash(key) % 10000) / 10000.0
    if bucket < test_frac:
        return "test"
    if bucket < test_frac + dev_frac:
        return "dev"
    return "train"


# --------------------------------------------------------------------------- validators
def validate_rag_query(row: Dict[str, Any]) -> List[str]:
    e: List[str] = []
    if not isinstance(row, dict):
        return ["query row is not an object"]
    for f in ("query_id", "query", "domain", "source"):
        if not (isinstance(row.get(f), str) and row[f].strip()):
            e.append(f"query missing '{f}'")
    pids = row.get("positive_doc_ids")
    if not isinstance(pids, list) or not pids or not all(isinstance(p, str) and p for p in pids):
        e.append("query 'positive_doc_ids' must be a non-empty list of strings")
    md = row.get("metadata")
    if md is not None:
        if not isinstance(md, dict):
            e.append("query 'metadata' must be an object")
        else:
            if "requires_answer_support" in md and not isinstance(md["requires_answer_support"], bool):
                e.append("metadata.requires_answer_support must be a bool")
            at = md.get("answer_type")
            if at is not None and at not in ANSWER_TYPES:
                e.append(f"metadata.answer_type must be one of {sorted(ANSWER_TYPES)}")
    return e


def validate_rag_corpus_doc(row: Dict[str, Any]) -> List[str]:
    e: List[str] = []
    if not isinstance(row, dict):
        return ["corpus row is not an object"]
    for f in ("doc_id", "text", "source", "domain", "license"):
        if not (isinstance(row.get(f), str) and row[f].strip()):
            e.append(f"corpus doc missing '{f}'")
    return e


def validate_candidate_list(row: Dict[str, Any], require_positive: bool = False) -> List[str]:
    e: List[str] = []
    if not isinstance(row, dict):
        return ["candidate-list row is not an object"]
    for f in ("query_id", "query"):
        if not (isinstance(row.get(f), str) and row[f].strip()):
            e.append(f"candidate list missing '{f}'")
    pids = set(row.get("positive_doc_ids") or [])
    cands = row.get("candidates")
    if not isinstance(cands, list) or not cands:
        e.append("candidate list 'candidates' must be a non-empty list")
        return e
    cand_ids = set()
    for c in cands:
        if not isinstance(c, dict) or not (isinstance(c.get("doc_id"), str) and c["doc_id"]):
            e.append("candidate missing 'doc_id'")
            continue
        cand_ids.add(c["doc_id"])
        if not isinstance(c.get("text"), str):
            e.append(f"candidate {c['doc_id']} missing 'text'")
        cs = c.get("candidate_source")
        if cs is not None and cs not in CANDIDATE_SOURCES:
            e.append(f"candidate {c['doc_id']}: candidate_source '{cs}' not in {sorted(CANDIDATE_SOURCES)}")
        if c.get("label") not in (0, 1, None):
            e.append(f"candidate {c['doc_id']}: label must be 1, 0, or null")
    if require_positive:
        has_pos = bool(pids & cand_ids) or any(c.get("label") == 1 for c in cands if isinstance(c, dict))
        if not has_pos:
            e.append("candidate list used for reranker lift must contain at least one positive")
    return e


def validate_eval_set(queries: Sequence[Dict[str, Any]], corpus: Sequence[Dict[str, Any]],
                      qrels: Optional[Sequence[Dict[str, Any]]] = None) -> List[str]:
    """Schema-validate queries+corpus and assert every positive_doc_id exists in the corpus."""
    errors: List[str] = []
    corpus_ids: Set[str] = set()
    for i, d in enumerate(corpus):
        errs = validate_rag_corpus_doc(d)
        errors += [f"corpus[{i}]: {x}" for x in errs]
        if not errs:
            corpus_ids.add(d["doc_id"])
    for i, q in enumerate(queries):
        errs = validate_rag_query(q)
        errors += [f"queries[{i}]: {x}" for x in errs]
        if not errs:
            for pid in q["positive_doc_ids"]:
                if pid not in corpus_ids:
                    errors.append(f"queries[{i}] ({q['query_id']}): positive_doc_id '{pid}' not in corpus")
    if qrels is not None:
        for i, r in enumerate(qrels):
            if not (isinstance(r.get("query_id"), str) and isinstance(r.get("doc_id"), str)):
                errors.append(f"qrels[{i}]: needs query_id + doc_id")
            elif r["doc_id"] not in corpus_ids:
                errors.append(f"qrels[{i}]: doc_id '{r['doc_id']}' not in corpus")
    return errors


def check_no_eval_leakage(train_rows: Sequence[Dict[str, Any]], eval_query_ids: Set[str],
                          eval_doc_ids: Set[str]) -> List[str]:
    """Public/eval data must never appear in training candidate files. Flags any train row whose
    query_id OR doc_id (or candidate doc_id) is in the eval split."""
    bad: List[str] = []
    for i, r in enumerate(train_rows):
        qid = str(r.get("query_id", ""))
        if qid in eval_query_ids:
            bad.append(f"train[{i}]: query_id '{qid}' is in the eval split")
        ids = {str(r.get("doc_id", ""))}
        for c in r.get("candidates", []) or []:
            if isinstance(c, dict):
                ids.add(str(c.get("doc_id", "")))
        for did in ids & eval_doc_ids:
            bad.append(f"train[{i}]: doc_id '{did}' is in the eval split")
    return bad


# ------------------------------------------------------------------ WebFAQ-style eval build
def build_webfaq_eval(faq_rows: Sequence[Dict[str, Any]], split: str = "test",
                      dev_frac: float = 0.1, test_frac: float = 0.1
                      ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic held-out eval from real FAQ rows ({query/question, document/answer, ...}).
    Returns (corpus, queries, qrels) for the requested split. The split is by stable hash of the
    query text, so train/dev/test never share a (query, answer) pair."""
    corpus, queries, qrels = [], [], []
    seen_docs = set()
    for r in faq_rows:
        q = (r.get("query") or r.get("question") or "").strip()
        a = (r.get("document") or r.get("answer") or "").strip()
        if not q or not a:
            continue
        if assign_split(q, dev_frac, test_frac) != split:
            continue
        qid = "q" + hashlib.blake2b(q.encode("utf-8"), digest_size=8).hexdigest()
        did = "d" + hashlib.blake2b(a.encode("utf-8"), digest_size=8).hexdigest()
        if did not in seen_docs:
            seen_docs.add(did)
            corpus.append({"doc_id": did, "text": a, "source": r.get("source", "webfaq"),
                           "domain": "faq_real", "license": r.get("license", "CC-BY-4.0"),
                           "url": r.get("url")})
        queries.append({"query_id": qid, "query": q, "answer": a, "positive_doc_ids": [did],
                        "domain": "faq_real", "source": r.get("source", "webfaq"),
                        "metadata": {"requires_answer_support": True, "answer_type": "faq"}})
        qrels.append({"query_id": qid, "doc_id": did, "relevance": 1})
    return corpus, queries, qrels


# ----------------------------------------------------------------------------- metrics
def rag_metrics_for_query(ranked_doc_ids: Sequence[str], positive_ids,
                          requires_answer_support: bool = False) -> Dict[str, Any]:
    """Per-query RAG metrics. Adds positive_in_top_10 and (when the query needs answer support)
    answer_support_at_10 to the standard nDCG/MRR/Recall@10."""
    positives = set(positive_ids)
    out = metrics_for_query(list(ranked_doc_ids), positives, (K,))
    top = set(ranked_doc_ids[:K])
    hit = 1.0 if (top & positives) else 0.0
    out["positive_in_top_10"] = hit
    if requires_answer_support:
        out["answer_support_at_10"] = hit       # only counted for answer-support queries
    return out


def aggregate_rag(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Mean over per-query metric dicts. answer_support_at_10 is averaged ONLY over the queries
    that carry it (answer-support queries)."""
    if not rows:
        return {}
    base_keys = ("ndcg@10", "mrr@10", "recall@10", "positive_in_top_10")
    out = {k: round(sum(r.get(k, 0.0) for r in rows) / len(rows), 4) for k in base_keys}
    asup = [r["answer_support_at_10"] for r in rows if "answer_support_at_10" in r]
    if asup:
        out["answer_support_at_10"] = round(sum(asup) / len(asup), 4)
        out["answer_support_queries"] = len(asup)
    return out


def reranker_delta_ndcg10(first_stage_ranked: Sequence[str], reranked: Sequence[str],
                          positive_ids) -> Dict[str, float]:
    """nDCG@10 of the first-stage order vs the reranked order over the SAME fixed candidate set."""
    positives = set(positive_ids)
    fs = metrics_for_query(list(first_stage_ranked), positives, (K,))["ndcg@10"]
    rr = metrics_for_query(list(reranked), positives, (K,))["ndcg@10"]
    return {"first_stage_ndcg@10": round(fs, 4), "reranked_ndcg@10": round(rr, 4),
            "reranker_delta_ndcg10": round(rr - fs, 4)}
