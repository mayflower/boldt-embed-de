"""Build a held-out NEAR-CEILING guardrail set for validating the bounded rerank policy WITHOUT
tuning on GermanQuAD/DT-test (pure stdlib, no ML).

A near-ceiling query = the first stage is already (almost) perfect, so raw reranking can only churn
it. Selecting such queries from non-public, leakage-safe, training-disjoint sources gives an
independent guardrail — GermanQuAD/DT-test stay guardrails, never the tuning target.

Near-ceiling definition (all required): first_stage_ndcg@10 >= 0.95, oracle_ndcg@10 >= 0.98,
positive_in_top_10, >= 20 candidates (>= 2 candidate sources preferred). Public-eval sources
(germanquad/dt_test) are excluded; any overlap with the training queries is a HARD failure.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Set

from .metrics import ndcg_at_k

K = 10
MIN_FIRST_STAGE_NDCG = 0.95
MIN_ORACLE_NDCG = 0.98
MIN_CANDIDATES = 20
MIN_SOURCES = 1                      # "at least 2 if possible" — soft; reported, not required

# TRUE public benchmarks excluded from the guardrail. NOTE: this deliberately omits "webfaq_heldout"
# — the held-out WebFAQ split is our own leakage-filtered guardrail SOURCE, not a public benchmark.
PUBLIC_EVAL_EXCLUDE = frozenset({"germanquad", "dt_test", "gerdalir", "germandpr",
                                 "miracl", "mldr", "sts22", "mmteb", "mteb"})


def _references_public_eval(text: str) -> bool:
    low = (text or "").lower()
    return any(tok in low for tok in PUBLIC_EVAL_EXCLUDE)


def _first_stage_order(cands):
    if cands and all(c.get("first_stage_rank") is not None for c in cands):
        return [c["doc_id"] for c in sorted(cands, key=lambda c: float(c["first_stage_rank"]))]
    if cands and all(c.get("first_stage_score") is not None for c in cands):
        return [c["doc_id"] for c in sorted(cands, key=lambda c: -float(c["first_stage_score"]))]
    return [c.get("doc_id") for c in cands]


def _positives(row) -> Set[str]:
    pos = set(row.get("positive_doc_ids") or [])
    if row.get("positive_doc_id"):
        pos.add(row["positive_doc_id"])
    pos |= {c.get("doc_id") for c in (row.get("candidates") or []) if c.get("is_positive")}
    return pos


def list_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    cands = row.get("candidates") or []
    fs = _first_stage_order(cands)
    pos = _positives(row)
    oracle = [d for d in fs if d in pos] + [d for d in fs if d not in pos]
    sources = sorted({c.get("candidate_source") for c in cands if c.get("candidate_source")})
    return {
        "query_id": row.get("query_id"), "domain": row.get("domain"),
        "first_stage_ndcg10": round(ndcg_at_k(fs, pos, K), 4),
        "oracle_ndcg10": round(ndcg_at_k(oracle, pos, K), 4),
        "positive_in_top_10": bool(set(fs[:K]) & pos),
        "num_candidates": len(cands), "num_candidate_sources": len(sources), "sources": sources,
    }


def is_near_ceiling(m: Dict[str, Any], *, min_fs: float = MIN_FIRST_STAGE_NDCG,
                    min_oracle: float = MIN_ORACLE_NDCG, min_candidates: int = MIN_CANDIDATES,
                    min_sources: int = MIN_SOURCES) -> bool:
    return (m["first_stage_ndcg10"] >= min_fs and m["oracle_ndcg10"] >= min_oracle
            and m["positive_in_top_10"] and m["num_candidates"] >= min_candidates
            and m["num_candidate_sources"] >= min_sources)


def is_excluded_source(row: Dict[str, Any], exclude: Set[str]) -> bool:
    """Exclude public-eval sources (germanquad/dt_test/…) and any explicitly excluded token."""
    fields = [str(row.get("domain") or ""), str(row.get("source") or ""),
              str(row.get("query_id") or "")]
    fields += [str(c.get("candidate_source") or "") for c in (row.get("candidates") or [])]
    low = " ".join(fields).lower()
    if any(tok and tok.lower() in low for tok in exclude):
        return True
    return any(_references_public_eval(f) for f in fields)


def _key(qid: str) -> str:
    return hashlib.blake2b(str(qid).encode("utf-8"), digest_size=8).hexdigest()


def _hist(values, edges):
    h = {f"[{edges[i]},{edges[i+1]})": 0 for i in range(len(edges) - 1)}
    h[f">={edges[-1]}"] = 0
    for v in values:
        placed = False
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                h[f"[{edges[i]},{edges[i+1]})"] += 1
                placed = True
                break
        if not placed and v >= edges[-1]:
            h[f">={edges[-1]}"] += 1
    return h


def build(rows: Sequence[Dict[str, Any]], *, exclude_sources: Optional[Set[str]] = None,
          train_query_ids: Optional[Set[str]] = None, target_size: int = 1000,
          min_fs: float = MIN_FIRST_STAGE_NDCG, min_oracle: float = MIN_ORACLE_NDCG,
          min_candidates: int = MIN_CANDIDATES, min_sources: int = MIN_SOURCES) -> Dict[str, Any]:
    exclude = {s.strip() for s in (exclude_sources or set()) if s and s.strip()}
    train_query_ids = train_query_ids or set()
    errors: List[str] = []

    excluded_public = 0
    pool: List[Dict[str, Any]] = []
    for r in rows:
        if is_excluded_source(r, exclude):
            excluded_public += 1
            continue
        m = list_metrics(r)
        if is_near_ceiling(m, min_fs=min_fs, min_oracle=min_oracle,
                           min_candidates=min_candidates, min_sources=min_sources):
            pool.append({"row": r, "m": m})

    # deterministic selection by stable query-id hash, then take target_size
    pool.sort(key=lambda x: (_key(x["m"]["query_id"]), str(x["m"]["query_id"])))
    selected = pool[:target_size]

    # HARD leakage guard: a guardrail must be disjoint from training.
    overlap = sorted({x["m"]["query_id"] for x in selected
                      if str(x["m"]["query_id"]) in {str(q) for q in train_query_ids}})
    if overlap:
        errors.append(f"{len(overlap)} selected query(ies) overlap training — guardrail must be "
                      f"train-disjoint; first: {overlap[0]}")

    metrics = [x["m"] for x in selected]
    by_domain: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    for m in metrics:
        by_domain[str(m["domain"])] = by_domain.get(str(m["domain"]), 0) + 1
        for s in m["sources"] or ["(none)"]:
            by_source[str(s)] = by_source.get(str(s), 0) + 1
    report = {
        "status": "fail" if errors else "pass",
        "errors": errors,
        "num_selected": len(selected),
        "target_size": target_size,
        "excluded_public_or_listed_source": excluded_public,
        "near_ceiling_definition": {"min_first_stage_ndcg10": min_fs, "min_oracle_ndcg10": min_oracle,
                                    "min_candidates": min_candidates, "min_sources": min_sources},
        "by_domain": dict(sorted(by_domain.items())),
        "candidate_source_distribution": dict(sorted(by_source.items())),
        "first_stage_ndcg_distribution": _hist([m["first_stage_ndcg10"] for m in metrics],
                                               [0.95, 0.97, 0.99, 1.0]),
        "oracle_ndcg_distribution": _hist([m["oracle_ndcg10"] for m in metrics], [0.98, 0.99, 1.0]),
        "multi_source_fraction": (round(sum(1 for m in metrics if m["num_candidate_sources"] >= 2)
                                        / len(metrics), 4) if metrics else 0.0),
        "leakage_check": {"public_eval_sources_excluded": excluded_public,
                          "exclude_tokens": sorted(exclude)},
        "training_overlap": {"checked_against_n_train_queries": len(train_query_ids),
                             "overlap_count": len(overlap), "overlap_query_ids": overlap[:50]},
    }
    return {"selected": [x["row"] for x in selected], "report": report, "errors": errors}
