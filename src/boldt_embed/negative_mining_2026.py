"""Teacher-driven, domain-balanced hard-negative mining (pure stdlib).

The v1 reranker failed because negatives were mined from a single domain by a weak model,
producing either too-easy negatives or *false* negatives (actually-relevant passages). This
module mines from multiple sources, then uses teacher scores to keep hard-but-plausible
negatives and drop likely false negatives, while balancing domains.

Pipeline:
  mine_bm25_candidates / mine_dense_candidates_from_embeddings  (multi-source pools)
    -> merge_candidate_pools                                    (union, source-labelled)
    -> filter_false_negatives                                   (teacher-score gated)
    -> select_domain_balanced_negatives                         (per-domain cap)
    -> build_triplets_or_lists                                  (final rows + stats)

All deterministic and dependency-free. Dense mining consumes *precomputed* embeddings, so
no model is loaded here. Output schema (one row per query):

    {"query_id","query","positive_doc_id","positive",
     "negatives":[{"doc_id","document","source","domain",
                   "embedding_teacher_score","reranker_teacher_score",
                   "false_negative_filter_reason"}],
     "source","domain"}
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .bm25_index import BM25Index, build_bm25_index
from .eval_harness import cosine_rank

ScoreKey = Tuple[str, str]


# ------------------------------------------------------------------- candidate pools
def mine_bm25_candidates(queries: Sequence[Dict[str, Any]], corpus: Sequence[Dict[str, Any]],
                         k: int = 50, index: Optional[BM25Index] = None) -> Dict[str, List[str]]:
    """Top-k BM25 doc ids per query. `queries`:[{query_id,query}], `corpus`:[{id,text}].

    The BM25 inverted index is built **once** over the corpus (or a prebuilt ``index`` is
    reused) and then queried per query — O(sum of query-term postings), not O(queries*corpus).
    This is the v3 fix for the v2 O(n*m) bottleneck that forced a ~3.5k mining subset."""
    if index is None:
        index = build_bm25_index(corpus)
    out: Dict[str, List[str]] = {}
    for q in queries:
        out[str(q["query_id"])] = [did for did, _ in index.search(q["query"], k)]
    return out


def mine_dense_candidates_from_embeddings(query_embeddings: Dict[str, Sequence[float]],
                                          doc_embeddings: Sequence[Tuple[str, Sequence[float]]],
                                          k: int = 50) -> Dict[str, List[str]]:
    """Top-k doc ids per query by cosine similarity over *precomputed* embeddings."""
    out: Dict[str, List[str]] = {}
    for qid, vec in query_embeddings.items():
        out[str(qid)] = cosine_rank(vec, doc_embeddings)[:k]
    return out


def merge_candidate_pools(*pools: Tuple[str, Dict[str, List[str]]]
                          ) -> Dict[str, List[Dict[str, str]]]:
    """Merge named pools into per-query ordered candidate lists, deduped by doc_id (first
    source that surfaced it wins). Each pool is ``(source_name, {qid: [doc_id, ...]})``."""
    merged: Dict[str, List[Dict[str, str]]] = {}
    seen: Dict[str, set] = {}
    for source_name, pool in pools:
        for qid, doc_ids in pool.items():
            merged.setdefault(qid, [])
            seen.setdefault(qid, set())
            for did in doc_ids:
                if did not in seen[qid]:
                    seen[qid].add(did)
                    merged[qid].append({"doc_id": did, "source": source_name})
    return merged


# ------------------------------------------------------------------- teacher scoring
def load_teacher_scores(cache_rows: Sequence[Dict[str, Any]]) -> Dict[ScoreKey, Dict[str, Any]]:
    """Index teacher-cache rows by (query_id, doc_id) -> {embedding_score, reranker_score}."""
    out: Dict[ScoreKey, Dict[str, Any]] = {}
    for r in cache_rows:
        out[(str(r["query_id"]), str(r["doc_id"]))] = {
            "embedding_score": r.get("embedding_score"),
            "reranker_score": r.get("reranker_score"),
        }
    return out


def _filter_score(scores: Optional[Dict[str, Any]]) -> Optional[float]:
    """The score used for false-negative comparison: prefer reranker, else embedding."""
    if not scores:
        return None
    if scores.get("reranker_score") is not None:
        return float(scores["reranker_score"])
    if scores.get("embedding_score") is not None:
        return float(scores["embedding_score"])
    return None


def false_negative_reason(pos_score: Optional[float], neg_score: Optional[float],
                          margin: float) -> Optional[str]:
    """Why this negative is a *likely false negative* (so should be dropped), or None to keep.

    If we cannot compare (a score is missing) we keep the negative — we never drop blindly."""
    if pos_score is None or neg_score is None:
        return None
    if neg_score >= pos_score:
        return "neg_score_ge_positive"
    if neg_score >= pos_score - margin:
        return "within_margin_of_positive"
    return None


def filter_false_negatives(pos_score: Optional[float], negatives: Sequence[Dict[str, Any]],
                           margin: float) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Split negatives into kept (reason=None) vs dropped, with per-reason counts."""
    kept: List[Dict[str, Any]] = []
    dropped: Dict[str, int] = {}
    for neg in negatives:
        reason = false_negative_reason(pos_score, _filter_score(neg.get("_scores")), margin)
        if reason:
            dropped[reason] = dropped.get(reason, 0) + 1
        else:
            neg = dict(neg)
            neg["false_negative_filter_reason"] = None
            kept.append(neg)
    return kept, dropped


# ------------------------------------------------------------------- selection/build
def select_domain_balanced_negatives(negatives: Sequence[Dict[str, Any]],
                                     max_per_domain: Optional[int]) -> List[Dict[str, Any]]:
    """Cap negatives per domain, preserving order. Deterministic."""
    if not max_per_domain:
        return list(negatives)
    counts: Dict[str, int] = {}
    out = []
    for n in negatives:
        dom = str(n.get("domain", "unknown"))
        if counts.get(dom, 0) < max_per_domain:
            out.append(n)
            counts[dom] = counts.get(dom, 0) + 1
    return out


def build_triplets_or_lists(positives: Sequence[Dict[str, Any]],
                            merged: Dict[str, List[Dict[str, str]]],
                            corpus_lookup: Dict[str, Dict[str, Any]],
                            teacher_scores: Dict[ScoreKey, Dict[str, Any]],
                            negatives_per_query: int = 8, margin: float = 0.1,
                            max_per_domain: Optional[int] = None
                            ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Assemble final hard-negative rows and a stats report.

    Negatives are scored by the teacher; likely false negatives are dropped; remaining are
    domain-balanced and the hardest ``negatives_per_query`` (highest teacher score below the
    positive) are kept. Returns (rows, stats)."""
    rows: List[Dict[str, Any]] = []
    stats = {"queries": 0, "total_candidates": 0, "kept": 0,
             "dropped_by_reason": {}, "kept_by_source": {}, "kept_by_domain": {}}
    for pos in positives:
        qid = str(pos["query_id"])
        pos_doc_id = str(pos["doc_id"])
        stats["queries"] += 1
        pos_score = _filter_score(teacher_scores.get((qid, pos_doc_id)))

        raw_negs = []
        for cand in merged.get(qid, []):
            did = cand["doc_id"]
            if did == pos_doc_id:
                continue
            doc = corpus_lookup.get(did, {})
            scores = teacher_scores.get((qid, did))
            raw_negs.append({
                "doc_id": did,
                "document": doc.get("text", doc.get("document", "")),
                "source": cand["source"],
                "domain": str(doc.get("domain", "unknown")),
                "embedding_teacher_score": (scores or {}).get("embedding_score"),
                "reranker_teacher_score": (scores or {}).get("reranker_score"),
                "_scores": scores,
            })
        stats["total_candidates"] += len(raw_negs)

        kept, dropped = filter_false_negatives(pos_score, raw_negs, margin)
        for reason, c in dropped.items():
            stats["dropped_by_reason"][reason] = stats["dropped_by_reason"].get(reason, 0) + c

        # Hardest first: highest teacher filter-score (closest to the positive), unscored
        # last; doc_id as a deterministic tie-break. Python's sort is stable, so sorting by
        # doc_id first then by score (desc) yields score-desc with doc_id-asc within ties.
        kept.sort(key=lambda n: n["doc_id"])
        kept.sort(key=lambda n: (_filter_score(n.get("_scores")) is not None,
                                 _filter_score(n.get("_scores")) or 0.0), reverse=True)

        kept = select_domain_balanced_negatives(kept, max_per_domain)
        kept = kept[:negatives_per_query]
        for n in kept:
            n.pop("_scores", None)
            stats["kept_by_source"][n["source"]] = stats["kept_by_source"].get(n["source"], 0) + 1
            stats["kept_by_domain"][n["domain"]] = stats["kept_by_domain"].get(n["domain"], 0) + 1
        stats["kept"] += len(kept)

        pos_doc = corpus_lookup.get(pos_doc_id, {})
        rows.append({
            "query_id": qid,
            "query": pos["query"],
            "positive_doc_id": pos_doc_id,
            "positive": pos.get("document") or pos_doc.get("text", pos_doc.get("document", "")),
            "negatives": kept,
            "source": pos.get("source", "unknown"),
            "domain": str(pos.get("domain", "unknown")),
        })
    stats["dropped_by_reason"] = dict(sorted(stats["dropped_by_reason"].items()))
    stats["kept_by_source"] = dict(sorted(stats["kept_by_source"].items()))
    stats["kept_by_domain"] = dict(sorted(stats["kept_by_domain"].items()))
    return rows, stats


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    return None if not xs else round(xs[len(xs) // 2], 4)


def build_reranker_candidate_lists(positives: Sequence[Dict[str, Any]],
                                   merged: Dict[str, List[Dict[str, str]]],
                                   corpus_lookup: Dict[str, Dict[str, Any]],
                                   teacher_scores: Dict[ScoreKey, Dict[str, Any]],
                                   negatives_per_query: int = 8, margin: float = 0.1
                                   ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build per-query CANDIDATE LISTS for reranker training (distribution-aware): each list is
    the positive (label 1) plus teacher-filtered, multi-source hard negatives (label 0), with
    ``teacher_score`` / ``candidate_source`` / ``domain`` per candidate. Returns (rows, stats).

    This is the v2 fix for the reranker generalization failure: candidates come from multiple
    sources (BM25/dense/...) so the reranker doesn't overfit one candidate distribution."""
    rows: List[Dict[str, Any]] = []
    stats = {"queries": 0, "positives": 0, "negatives": 0, "vetoed_false_negatives": 0,
             "candidates_by_source": {}, "candidates_by_domain": {},
             "pos_teacher_median": None, "neg_teacher_median": None}
    pos_scores_all, neg_scores_all = [], []
    for pos in positives:
        qid = str(pos["query_id"]); pos_doc_id = str(pos["doc_id"])
        stats["queries"] += 1
        pos_score = _filter_score(teacher_scores.get((qid, pos_doc_id)))
        pos_scores_all.append(pos_score)
        pos_doc = corpus_lookup.get(pos_doc_id, {})
        cands = [{"doc_id": pos_doc_id,
                  "document": pos.get("document") or pos_doc.get("text", pos_doc.get("document", "")),
                  "label": 1, "teacher_score": pos_score, "first_stage_score": None,
                  "candidate_source": "positive", "domain": str(pos.get("domain", "unknown"))}]
        stats["positives"] += 1

        kept = []
        for cand in merged.get(qid, []):
            did = cand["doc_id"]
            if did == pos_doc_id:
                continue
            scores = teacher_scores.get((qid, did))
            ns = _filter_score(scores)
            if false_negative_reason(pos_score, ns, margin):
                stats["vetoed_false_negatives"] += 1
                continue
            doc = corpus_lookup.get(did, {})
            kept.append({"doc_id": did,
                         "document": doc.get("text", doc.get("document", "")),
                         "label": 0, "teacher_score": ns, "first_stage_score": None,
                         "candidate_source": cand["source"],
                         "domain": str(doc.get("domain", "unknown")), "_s": ns})
        kept.sort(key=lambda c: c["doc_id"])
        kept.sort(key=lambda c: (c["_s"] is not None, c["_s"] or 0.0), reverse=True)
        kept = kept[:negatives_per_query]
        for c in kept:
            c.pop("_s", None)
            neg_scores_all.append(c["teacher_score"])
            stats["candidates_by_source"][c["candidate_source"]] = \
                stats["candidates_by_source"].get(c["candidate_source"], 0) + 1
            stats["candidates_by_domain"][c["domain"]] = \
                stats["candidates_by_domain"].get(c["domain"], 0) + 1
            stats["negatives"] += 1
        rows.append({"query_id": qid, "query": pos["query"], "candidates": cands + kept,
                     "positive_doc_ids": [pos_doc_id], "source": pos.get("source", "unknown"),
                     "domain": str(pos.get("domain", "unknown"))})
    stats["candidates_by_source"] = dict(sorted(stats["candidates_by_source"].items()))
    stats["candidates_by_domain"] = dict(sorted(stats["candidates_by_domain"].items()))
    stats["pos_teacher_median"] = _median(pos_scores_all)
    stats["neg_teacher_median"] = _median(neg_scores_all)
    return rows, stats
