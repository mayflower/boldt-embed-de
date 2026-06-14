"""Build realistic fixed top-k candidate lists for RAG reranker training/eval (pure stdlib).

A reranker should reorder a real first-stage top-k, not classify random pairs. This merges the
top-k from several first stages (BM25 + v3 dense + e5 + qwen + WebFAQ hard negatives), preserves
each candidate's source, deduplicates by doc_id and text hash, labels (train: high-precision
teacher labels with uncertain=null; eval: labels stay null — positives come from qrels), ensures
eval lists carry a positive (or reports it), and emits a first-stage-recall report.

No ML, no network.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .reranker_modern import v3_label

# Merge/dedup priority: the first source that surfaced a doc wins its candidate_source tag.
SOURCE_PRIORITY = ("bm25", "v3_dense", "e5_dense", "qwen_dense", "webfaq_hardneg", "manual")


def _text_hash(text: str) -> str:
    return hashlib.blake2b(" ".join(str(text).lower().split()).encode("utf-8"),
                           digest_size=12).hexdigest()


def _ranked_ids(results_for_q: Sequence[Any]) -> List[Tuple[str, Optional[float]]]:
    """Normalize a per-query result list into [(doc_id, score?)] in rank order."""
    out = []
    for it in results_for_q or []:
        if isinstance(it, dict):
            out.append((str(it.get("doc_id") or it.get("id")), it.get("score")))
        else:
            out.append((str(it), None))
    return out


def build_candidate_lists(queries: Sequence[Dict[str, Any]], corpus: Dict[str, Dict[str, Any]],
                          source_results: Dict[str, Dict[str, Sequence[Any]]], *,
                          positives_by_query: Optional[Dict[str, Set[str]]] = None,
                          teacher_scores: Optional[Dict[Tuple[str, str], float]] = None,
                          hard_negatives: Optional[Dict[str, Sequence[str]]] = None,
                          top_k: int = 20, is_eval: bool = False,
                          positive_threshold: float = 4.0, neg_margin: float = 2.0
                          ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    teacher_scores = teacher_scores or {}
    hard_negatives = hard_negatives or {}
    positives_by_query = positives_by_query or {}
    # order sources by the merge priority (known first, then any extras deterministically)
    src_order = [s for s in SOURCE_PRIORITY if s in source_results] + \
                sorted(s for s in source_results if s not in SOURCE_PRIORITY)

    rows: List[Dict[str, Any]] = []
    missing_positive: List[str] = []
    injected_positive: List[str] = []
    src_dist: Dict[str, int] = {}
    pos_in_topk = 0
    counts: List[int] = []

    for q in queries:
        qid = str(q["query_id"])
        positives = set(q.get("positive_doc_ids") or []) | set(positives_by_query.get(qid, set()))
        seen_ids: Set[str] = set()
        seen_text: Set[str] = set()
        cands: List[Dict[str, Any]] = []

        def _add(doc_id: str, source: str, rank: Optional[int], score: Optional[float]):
            if doc_id in seen_ids or doc_id not in corpus:
                return
            text = corpus[doc_id].get("text") or corpus[doc_id].get("document") or ""
            th = _text_hash(text)
            if th in seen_text:           # same text under a different id -> dedup
                return
            seen_ids.add(doc_id)
            seen_text.add(th)
            ts = teacher_scores.get((qid, doc_id))
            if is_eval:
                label = None              # eval: positives come from qrels, not fabricated labels
            elif doc_id in positives:
                label = 1                 # gold positive -> high-precision positive
            else:
                label = v3_label(ts, positive_threshold, neg_margin)  # teacher label or null
            cands.append({"doc_id": doc_id, "text": text, "candidate_source": source,
                          "first_stage_rank": rank,
                          "first_stage_score": float(score) if score is not None else None,
                          "teacher_score": float(ts) if ts is not None else None,
                          "label": label, "domain": corpus[doc_id].get("domain", q.get("domain", "unknown"))})

        for source in src_order:
            for rank, (did, score) in enumerate(_ranked_ids(source_results[source].get(qid, []))[:top_k]):
                _add(did, source, rank, score)
        for did in hard_negatives.get(qid, []):       # train-time WebFAQ hard negatives
            _add(str(did), "webfaq_hardneg", None, None)

        present = positives & seen_ids
        if present:
            pos_in_topk += 1
        else:
            missing_positive.append(qid)
            if is_eval:
                # inject the gold positive so the eval list is scorable for lift
                for pid in sorted(positives):
                    if pid in corpus:
                        _add(pid, "manual", None, None)
                        injected_positive.append(qid)
                        break
        if not is_eval and not present:
            continue   # train: no positive -> cannot form a list; skipped (counted in missing)

        for c in cands:
            src_dist[c["candidate_source"]] = src_dist.get(c["candidate_source"], 0) + 1
        counts.append(len(cands))
        rows.append({"query_id": qid, "query": q.get("query", ""),
                     "positive_doc_ids": sorted(positives), "candidates": cands,
                     "domain": q.get("domain", "unknown"), "source": q.get("source", "unknown")})

    n = len(rows)
    report = {
        "mode": "eval" if is_eval else "train",
        "n_queries_in": len(queries), "n_lists_out": n,
        "candidates_per_query": {
            "min": min(counts) if counts else 0, "max": max(counts) if counts else 0,
            "mean": round(sum(counts) / n, 2) if n else 0.0},
        "positive_in_top_k_rate": round(pos_in_topk / len(queries), 4) if queries else 0.0,
        "candidate_source_distribution": dict(sorted(src_dist.items())),
        "domains": _count_key(rows, "domain"),
        "missing_positive_queries": missing_positive,
        "n_missing_positive": len(missing_positive),
        "injected_positive_queries": injected_positive,
        "top_k": top_k,
    }
    return rows, report


def _count_key(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        out[str(r.get(key, "unknown"))] = out.get(str(r.get(key, "unknown")), 0) + 1
    return dict(sorted(out.items()))
