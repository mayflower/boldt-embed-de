"""Catastrophic-drop error analysis for the v5 reranker (pure stdlib, no ML).

For each GermanQuAD query where reranking (always-rerank over the conservative scores) catastrophically
drops nDCG@10 vs the first stage, build a structured record, classify the error type, and check
which bounded policy would fix it. Answers: are the remaining failures POLICY-fixable (bounded
reranking) or DATA/MODEL-fixable (the reranker genuinely prefers a wrong doc)?

Uses first-stage scores/ranks, reranker scores, candidate text, and qrels (qrels for ANALYSIS only).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .bounded_rerank import _first_stage, _reranker, apply_policy
from .metrics import ndcg_at_k
from .rerank_abstain import extract_features

K = 10
CATASTROPHIC_DROP = 0.2
_DID, _TXT, _RR, _FSS, _SRC = ("doc_id", "text", "reranker_score", "first_stage_score",
                               "candidate_source")
ERROR_TYPES = ("positive_demoted_from_top1_or_top3", "lexical_exact_positive_demoted",
               "reranker_promotes_longer_but_wrong_doc", "duplicate_or_near_duplicate_confusion",
               "query_style_mismatch", "candidate_source_artifact",
               "insufficient_first_stage_features", "unknown")
FIX_POLICIES = (("top1_lock", "top1_lock", {}), ("top3_lock", "topk_lock", {"k": 3}),
                ("bounded_downshift_D1", "bounded_downshift", {"D": 1}),
                ("blend_alpha0.85", "blend", {"alpha": 0.85}),
                ("margin_override_M3", "margin_override", {"margin": 3.0}))


def _positives(row):
    pos = set(row.get("positive_doc_ids") or [])
    if row.get("positive_doc_id"):
        pos.add(row["positive_doc_id"])
    pos |= {c.get(_DID) for c in (row.get("candidates") or []) if c.get("is_positive")}
    return pos


def _tokens(t: str):
    return set((t or "").lower().split())


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def classify_error(row: Dict[str, Any]) -> str:
    cands = row.get("candidates") or []
    positives = _positives(row)
    fs = [c.get(_DID) for c in _first_stage(cands)]
    rr = [c.get(_DID) for c in _reranker(cands)]
    by_id = {c.get(_DID): c for c in cands}
    if not positives or not fs:
        return "unknown"
    pos_fs = min((fs.index(p) for p in positives if p in fs), default=len(fs))
    pos_rr = min((rr.index(p) for p in positives if p in rr), default=len(rr))
    best_pos = min((p for p in positives if p in fs), key=lambda p: fs.index(p), default=None)

    # 1/2: a positive that the first stage had near the top got demoted out of the top-K
    if pos_fs <= 2 and pos_rr >= K:
        if pos_fs == 0:
            return "lexical_exact_positive_demoted"
        return "positive_demoted_from_top1_or_top3"
    # 3: reranker put a NON-positive, notably-longer doc at the top
    rr_top1 = by_id.get(rr[0], {})
    if best_pos is not None and rr[0] not in positives:
        if len(rr_top1.get(_TXT, "")) > 1.3 * max(1, len(by_id.get(best_pos, {}).get(_TXT, ""))):
            return "reranker_promotes_longer_but_wrong_doc"
    # 4: a non-positive near-duplicate of the positive sits in the candidate list
    if best_pos is not None:
        ptok = _tokens(by_id.get(best_pos, {}).get(_TXT, ""))
        for c in cands:
            if c.get(_DID) not in positives and _jaccard(ptok, _tokens(c.get(_TXT, ""))) >= 0.8:
                return "duplicate_or_near_duplicate_confusion"
    feat = extract_features(row)
    # 6: genuine multi-source artifact — reranker promoted a wrong doc from a DIFFERENT source than
    #    the positive (only meaningful when >= 2 candidate sources are present).
    if (best_pos is not None and feat["num_candidate_sources"] >= 2 and rr[0] not in positives
            and by_id.get(rr[0], {}).get(_SRC) != by_id.get(best_pos, {}).get(_SRC)):
        return "candidate_source_artifact"
    # 7: first stage couldn't separate the top candidates (tiny gap) -> ambiguous, not the reranker's
    if feat["first_stage_top1_top2_gap"] < 0.5:
        return "insufficient_first_stage_features"
    # 5: the positive passage is far longer than the (short) query
    q_words = len((row.get("query") or "").split())
    pos_words = len(by_id.get(best_pos, {}).get(_TXT, "").split()) if best_pos else 0
    if q_words and pos_words and (pos_words > 8 * q_words):
        return "query_style_mismatch"
    return "unknown"


def _ndcg(order, positives):
    return ndcg_at_k(order, positives, K)


def fixability(row: Dict[str, Any], first_stage_ndcg: float) -> Dict[str, bool]:
    """Which bounded policy turns this catastrophic drop into a non-catastrophic one."""
    positives = _positives(row)
    out = {}
    for label, policy, params in FIX_POLICIES:
        order, _ = apply_policy(row, policy, params)
        out[label] = (_ndcg(order, positives) - first_stage_ndcg) > -CATASTROPHIC_DROP + 1e-9
    return out


def analyze_query(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cands = row.get("candidates") or []
    if len(cands) < 2:
        return None
    positives = _positives(row)
    fs = [c.get(_DID) for c in _first_stage(cands)]
    rr = [c.get(_DID) for c in _reranker(cands)]
    fs_rank = {d: i for i, d in enumerate(fs)}
    fsn = _ndcg(fs, positives)
    rrn = _ndcg(rr, positives)
    delta = rrn - fsn
    if delta > -CATASTROPHIC_DROP:
        return None                                   # not a catastrophic drop
    feat = extract_features(row)
    src_mix: Dict[str, int] = {}
    for c in cands:
        src_mix[str(c.get(_SRC))] = src_mix.get(str(c.get(_SRC)), 0) + 1
    return {
        "query_id": row.get("query_id"), "query": row.get("query"),
        "first_stage_ndcg10": round(fsn, 4), "reranked_ndcg10": round(rrn, 4),
        "delta": round(delta, 4),
        "first_stage_top10": fs[:K], "reranked_top10": rr[:K],
        "positive_doc_ids": sorted(positives),
        "positive_initial_rank": min((fs_rank[p] for p in positives if p in fs_rank), default=None),
        "positive_final_rank": min((rr.index(p) for p in positives if p in rr), default=None),
        "first_stage_gap_features": {k: feat[k] for k in
                                     ("first_stage_top1_top2_gap", "first_stage_top1_top5_gap",
                                      "first_stage_score_entropy")},
        "reranker_gap_features": {k: feat[k] for k in
                                  ("reranker_top1_top2_gap", "reranker_score_entropy")},
        "rank_displacements": [rr.index(d) - fs_rank[d] for d in fs],
        "candidate_source_mix": dict(sorted(src_mix.items())),
        "error_type": classify_error(row),
        "fixable_by": fixability(row, fsn),
    }


def analyze(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    records = [r for r in (analyze_query(x) for x in rows) if r]
    records.sort(key=lambda r: r["delta"])              # most negative first; deterministic
    n = len(records)
    by_type: Dict[str, int] = {t: 0 for t in ERROR_TYPES}
    fix_counts: Dict[str, int] = {label: 0 for label, _, _ in FIX_POLICIES}
    gaps_fs, gaps_rr = [], []
    for r in records:
        by_type[r["error_type"]] = by_type.get(r["error_type"], 0) + 1
        for label in fix_counts:
            if r["fixable_by"].get(label):
                fix_counts[label] += 1
        gaps_fs.append(r["first_stage_gap_features"]["first_stage_top1_top2_gap"])
        gaps_rr.append(r["reranker_gap_features"]["reranker_top1_top2_gap"])

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0
    any_fix = sum(1 for r in records if any(r["fixable_by"].values()))
    return {
        "n_queries_total": len(rows), "n_catastrophic": n,
        "counts_by_error_type": {k: v for k, v in by_type.items() if v} or by_type,
        "avg_first_stage_gap": mean(gaps_fs), "avg_reranker_gap": mean(gaps_rr),
        "fixable_counts": fix_counts,
        "fixable_by_any_policy": any_fix,
        "not_fixable_by_any_policy": n - any_fix,
        "policy_fixable_fraction": round(any_fix / n, 4) if n else 0.0,
        "records": records,
    }


def render_markdown(report: Dict[str, Any], top: int = 20) -> str:
    L = ["# v5 catastrophic GermanQuAD drop analysis", "",
         f"- catastrophic queries: **{report['n_catastrophic']}** of {report['n_queries_total']}",
         f"- avg first-stage gap: {report['avg_first_stage_gap']}; "
         f"avg reranker gap: {report['avg_reranker_gap']}",
         f"- **policy-fixable (any bounded policy): {report['fixable_by_any_policy']} "
         f"({report['policy_fixable_fraction']*100:.1f}%)**; "
         f"data/model-fixable (none): {report['not_fixable_by_any_policy']}", "",
         "## Counts by error type", "", "| error_type | count |", "|---|--:|"]
    for k, v in report["counts_by_error_type"].items():
        L.append(f"| {k} | {v} |")
    L += ["", "## Fixed by policy (of the catastrophic drops)", "", "| policy | fixes |", "|---|--:|"]
    for label, c in report["fixable_counts"].items():
        L.append(f"| {label} | {c} |")
    L += ["", f"## Top {top} catastrophic examples", "",
          "| query_id | fs→rr nDCG | Δ | pos init→final rank | error_type | fixed_by |",
          "|---|---|--:|---|---|---|"]
    for r in report["records"][:top]:
        fixed = ",".join(k for k, v in r["fixable_by"].items() if v) or "none"
        L.append(f"| {r['query_id']} | {r['first_stage_ndcg10']}→{r['reranked_ndcg10']} | "
                 f"{r['delta']:+} | {r['positive_initial_rank']}→{r['positive_final_rank']} | "
                 f"{r['error_type']} | {fixed} |")
    return "\n".join(L) + "\n"
