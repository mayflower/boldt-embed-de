"""Audit FIRST-STAGE retrieval recall before training another reranker (pure stdlib, no ML).

A reranker can only reorder documents the first stage already retrieved. If a positive is absent
from the candidate list, NO reranker — raw or policy-gated — can recover it. This module separates
the two bottlenecks:

  * RECALL bottleneck  — the positive was never retrieved (only present via an injected/oracle
    source such as ``manual``, or absent entirely). Fix = better retrieval / candidate construction.
  * RERANKER bottleneck — the positive WAS retrieved but ranked low. Fix = reranker quality.

Crucial honesty rule: injected/oracle candidate sources (``manual``/``gold``/…) are NOT retriever
hits. Recall and the realistic "perfect-reranker" ceiling are computed over the RETRIEVER set only;
counting injected positives would overstate what any reranker can do.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Set

from .metrics import ndcg_at_k

K = 10
DEFAULT_KS = (10, 20, 50, 100, 200)

# candidate sources that are NOT real retrievers — gold injected for training/eval, never a hit.
INJECTED_SOURCES = frozenset({"manual", "gold", "injected", "oracle", "positive", "label",
                              "qrel", "groundtruth", "ground_truth"})
DENSE_SOURCE_TOKENS = ("dense", "e5", "bge", "qwen", "embed", "boldt", "gte", "minilm")
BM25_SOURCE_TOKENS = ("bm25", "bm-25", "lexical", "sparse", "tfidf", "tf-idf")


def classify_source(src: Optional[str]) -> str:
    """-> 'injected' | 'bm25' | 'dense' | 'other'."""
    s = (src or "").strip().lower()
    if not s:
        return "other"
    if s in INJECTED_SOURCES:
        return "injected"
    if any(t in s for t in BM25_SOURCE_TOKENS):
        return "bm25"
    if any(t in s for t in DENSE_SOURCE_TOKENS):
        return "dense"
    return "other"


def _positives(row: Dict[str, Any], qrels: Optional[Dict[str, Set[str]]] = None) -> Set[str]:
    qid = str(row.get("query_id"))
    if qrels and qid in qrels:
        return set(qrels[qid])
    pos = set(row.get("positive_doc_ids") or [])
    if row.get("positive_doc_id"):
        pos.add(row["positive_doc_id"])
    pos |= {c.get("doc_id") for c in (row.get("candidates") or []) if c.get("is_positive")}
    pos |= {c.get("doc_id") for c in (row.get("candidates") or []) if c.get("label") == 1}
    return {p for p in pos if p}


def _fs_key(c: Dict[str, Any]):
    r = c.get("first_stage_rank")
    s = c.get("first_stage_score")
    return (r is None, float(r) if r is not None else 0.0,
            -(float(s) if s is not None else float("-inf")), str(c.get("doc_id")))


def _retriever_order(cands: Sequence[Dict[str, Any]], injected: Set[str]) -> List[str]:
    """First-stage ranking restricted to RETRIEVER candidates (injected sources excluded)."""
    retr = [c for c in cands if classify_source(c.get("candidate_source")) != "injected"
            and c.get("candidate_source", "").lower() not in injected]
    return [c["doc_id"] for c in sorted(retr, key=_fs_key)]


def _full_order(cands: Sequence[Dict[str, Any]]) -> List[str]:
    return [c["doc_id"] for c in sorted(cands, key=_fs_key)]


def _oracle(ranked_ids: Sequence[str], positives: Set[str]) -> List[str]:
    present_pos = [d for d in ranked_ids if d in positives]
    rest = [d for d in ranked_ids if d not in positives]
    return present_pos + rest


def audit_query(row: Dict[str, Any], *, ks: Sequence[int] = DEFAULT_KS,
                injected: Set[str] = INJECTED_SOURCES,
                qrels: Optional[Dict[str, Set[str]]] = None) -> Dict[str, Any]:
    cands = row.get("candidates") or []
    by_id = {c["doc_id"]: c for c in cands}
    pos = _positives(row, qrels)
    retr_order = _retriever_order(cands, set(injected))   # retriever-only first-stage order
    full_order = _full_order(cands)                       # incl injected (what the reranker sees)
    retr_set = set(retr_order)

    def recall(k):
        return (len(pos & set(retr_order[:k])) / len(pos)) if pos else 0.0
    recalls = {f"recall@{k}": round(recall(k), 6) for k in ks}

    fs_ndcg = ndcg_at_k(retr_order, pos, K)                       # actual first-stage (retriever)
    oracle_retr = ndcg_at_k(_oracle(retr_order, pos), pos, K)     # perfect rerank of RETRIEVER set
    oracle_full = ndcg_at_k(_oracle(full_order, pos), pos, K)     # perfect rerank incl injected

    retrieved = {p for p in pos if p in retr_set}
    missing = sorted(pos - retrieved)                            # positive never retrieved
    # which class supplied each positive (retriever hit vs injected-only vs absent)
    def classes_of(doc):
        return {classify_source(c.get("candidate_source")) for c in cands if c["doc_id"] == doc}
    pos_bm25 = {p for p in pos if "bm25" in classes_of(p)}
    pos_dense = {p for p in pos if "dense" in classes_of(p)}
    pos_other_retr = {p for p in pos if "other" in classes_of(p)}
    pos_injected_only = {p for p in pos if p in by_id and p not in retrieved}
    pos_absent = {p for p in pos if p not in by_id}
    return {
        "query_id": row.get("query_id"), "domain": row.get("domain"),
        "query_style": row.get("query_style") or row.get("domain") or row.get("source"),
        "num_positives": len(pos), **recalls,
        "positive_in_top_10": bool(pos & set(retr_order[:K])),
        "first_stage_ndcg10": round(fs_ndcg, 6),
        "oracle_ndcg10_retriever": round(oracle_retr, 6),
        "oracle_ndcg10_with_injected": round(oracle_full, 6),
        "upper_bound_reranker_lift": round(oracle_retr - fs_ndcg, 6),     # realistic ceiling
        "illusory_lift_from_injection": round(oracle_full - oracle_retr, 6),
        "retrieved_positives": sorted(retrieved), "missing_positives": missing,
        "missing_count": len(missing),
        "pos_bm25": sorted(pos_bm25), "pos_dense": sorted(pos_dense),
        "pos_other_retriever": sorted(pos_other_retr),
        "pos_injected_only": sorted(pos_injected_only), "pos_absent": sorted(pos_absent),
    }


def _key(qid: str) -> str:
    return hashlib.blake2b(str(qid).encode("utf-8"), digest_size=8).hexdigest()


def audit_set(rows: Sequence[Dict[str, Any]], *, name: str = "?", ks: Sequence[int] = DEFAULT_KS,
              injected: Set[str] = INJECTED_SOURCES, qrels: Optional[Dict[str, Set[str]]] = None,
              corpus: Optional[Dict[str, str]] = None, queries: Optional[Dict[str, str]] = None,
              max_examples: int = 8) -> Dict[str, Any]:
    per = [audit_query(r, ks=ks, injected=injected, qrels=qrels) for r in rows
           if (r.get("candidates") or [])]
    n = len(per) or 1
    tot_pos = sum(p["num_positives"] for p in per) or 1

    def mean(key):
        return round(sum(p[key] for p in per) / len(per), 6) if per else 0.0
    # candidate-source contribution over ALL positives (set arithmetic per positive)
    bm25 = dense = other = inj_only = absent = both = union = 0
    by_domain_missing: Dict[str, int] = {}
    by_style_missing: Dict[str, int] = {}
    by_source_of_missing: Dict[str, int] = {}
    for p in per:
        b = bool(p["pos_bm25"]); d = bool(p["pos_dense"]); o = bool(p["pos_other_retriever"])
        bm25 += len(p["pos_bm25"]); dense += len(p["pos_dense"]); other += len(p["pos_other_retriever"])
        inj_only += len(p["pos_injected_only"]); absent += len(p["pos_absent"])
        both += len(set(p["pos_bm25"]) & set(p["pos_dense"]))
        union += len(set(p["pos_bm25"]) | set(p["pos_dense"]) | set(p["pos_other_retriever"]))
        if p["missing_count"]:
            by_domain_missing[str(p["domain"])] = by_domain_missing.get(str(p["domain"]), 0) + p["missing_count"]
            by_style_missing[str(p["query_style"])] = by_style_missing.get(str(p["query_style"]), 0) + p["missing_count"]
            # what source DID hold the missing positive (e.g. injected 'manual', or absent)
            tag = "absent" if p["pos_absent"] else "injected_only"
            by_source_of_missing[tag] = by_source_of_missing.get(tag, 0) + p["missing_count"]
    bm25_only = bm25 - both
    dense_only = dense - both

    retriever_missing = inj_only + absent           # reranker can NEVER reach these
    missing_rate = round(retriever_missing / tot_pos, 6)
    fs = mean("first_stage_ndcg10")
    oracle_retr = mean("oracle_ndcg10_retriever")
    oracle_full = mean("oracle_ndcg10_with_injected")

    # deterministic missing examples
    missing_rows = sorted([p for p in per if p["missing_count"]],
                          key=lambda p: (_key(p["query_id"]), str(p["query_id"])))
    examples = []
    for p in missing_rows[:max_examples]:
        mp = p["missing_positives"][0]
        examples.append({
            "query_id": p["query_id"], "domain": p["domain"], "query_style": p["query_style"],
            "query": (queries or {}).get(str(p["query_id"]), ""),
            "missing_positive": mp,
            "missing_positive_text": ((corpus or {}).get(mp, "") or "")[:200],
            "first_stage_ndcg10": p["first_stage_ndcg10"],
            "oracle_ndcg10_retriever": p["oracle_ndcg10_retriever"],
            "note": "positive present only via injected/oracle source — first stage never retrieved it"
                    if p["pos_injected_only"] else "positive absent from candidate list entirely",
        })

    bottleneck = _classify_bottleneck(missing_rate, fs, oracle_retr, dense, bm25)
    return {
        "eval_set": name, "n_queries": len(per), "total_positives": tot_pos,
        "recall": {f"recall@{k}": mean(f"recall@{k}") for k in ks},
        "positive_in_top_10_rate": round(sum(1 for p in per if p["positive_in_top_10"]) / n, 6),
        "first_stage_ndcg10": fs,
        "oracle_ndcg10_retriever": oracle_retr,
        "oracle_ndcg10_with_injected": oracle_full,
        "upper_bound_reranker_lift_realistic": round(oracle_retr - fs, 6),
        "upper_bound_reranker_lift_if_perfect_candidates": round(oracle_full - fs, 6),
        "illusory_lift_from_injection": round(oracle_full - oracle_retr, 6),
        "missing_positive_count": retriever_missing,
        "missing_positive_rate": missing_rate,
        "missing_by_domain": dict(sorted(by_domain_missing.items(), key=lambda kv: -kv[1])),
        "missing_by_query_style": dict(sorted(by_style_missing.items(), key=lambda kv: -kv[1])),
        "missing_by_source": dict(sorted(by_source_of_missing.items(), key=lambda kv: -kv[1])),
        "candidate_source_contribution": {
            "total_positives": tot_pos, "bm25_hits": bm25, "dense_hits": dense,
            "other_retriever_hits": other, "bm25_only": bm25_only, "dense_only": dense_only,
            "overlap_bm25_and_dense": both, "union_any_retriever": union,
            "injected_only_not_retrieved": inj_only, "absent_from_list": absent,
            "has_dense_source": dense > 0,
        },
        "bottleneck": bottleneck,
        "examples": examples,
    }


def _classify_bottleneck(missing_rate, fs_ndcg, oracle_retr, dense_hits, bm25_hits) -> Dict[str, Any]:
    realistic_lift = round(oracle_retr - fs_ndcg, 6)
    if missing_rate >= 0.10:
        primary = "first_stage_recall"
        detail = (f"{missing_rate*100:.1f}% of positives were never retrieved — the reranker cannot "
                  "see them. Fix retrieval / candidate-list construction, not the reranker.")
    elif fs_ndcg >= 0.95:
        primary = "near_ceiling_first_stage"
        detail = ("first stage is already near-ceiling; reranking can only churn it. Reranker lift is "
                  "not the lever here.")
    elif realistic_lift >= 0.05:
        primary = "reranker_quality"
        detail = ("positives are retrieved but ranked low; a better reranker could realistically lift "
                  f"nDCG@10 by up to {realistic_lift}.")
    else:
        primary = "limited_headroom"
        detail = "positives mostly retrieved and reasonably ranked; little realistic reranker headroom."
    return {"primary": primary, "detail": detail,
            "realistic_reranker_ceiling": realistic_lift,
            "no_dense_first_stage": dense_hits == 0 and bm25_hits > 0}
