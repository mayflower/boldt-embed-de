"""Conservative rerank-or-abstain policy for the v5 reranker (pure stdlib, no ML, no training).

The v5 reranker helps where there is headroom (medium+hard) but churns near-ceiling first-stage
lists (GermanQuAD overall -0.0285, 16.9% catastrophic). This module wraps the EXISTING reranker
scores with a production-feasible policy: abstain (keep the first stage) on confident/near-ceiling
lists, rerank only uncertain ones.

INFERENCE USES OBSERVABLE FEATURES ONLY — never qrels/labels/teacher_score/oracle_ndcg/
hardness_bucket. Those may be used ONLY for fitting (on a dev split) and for eval/analysis.
``extract_features`` and ``apply_policy`` read only first_stage_score/first_stage_rank/
reranker_score/candidate_source/doc_id.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .metrics import ndcg_at_k

K = 10
CATASTROPHIC_DROP = 0.2          # per-query policy-vs-first-stage nDCG@10 drop counted catastrophic

ACTIONS = ("keep_first_stage", "rerank_top_k", "conservative_blend", "rerank_only_if_margin")
POLICIES = ("always_rerank", "never_rerank", "first_stage_confidence_abstain",
            "reranker_confidence_gate", "displacement_guard", "conservative_blend",
            "combined_policy")

# ---- inference-safe fields only (anything not here must NOT be read at inference) ----
_FS_SCORE, _FS_RANK, _RR_SCORE, _SRC, _DID = (
    "first_stage_score", "first_stage_rank", "reranker_score", "candidate_source", "doc_id")


def _fnum(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _first_stage_order(cands: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if cands and all(c.get(_FS_RANK) is not None for c in cands):
        return sorted(cands, key=lambda c: _fnum(c[_FS_RANK], 1e9))
    if cands and all(c.get(_FS_SCORE) is not None for c in cands):
        return sorted(cands, key=lambda c: -_fnum(c[_FS_SCORE]))
    return list(cands)


def _reranker_order(cands: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(cands, key=lambda c: -_fnum(c.get(_RR_SCORE)))


def _softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    e = [math.exp(v - m) for v in xs]
    s = sum(e) or 1.0
    return [v / s for v in e]


def _entropy(scores: List[float]) -> float:
    p = _softmax(scores)
    return round(-sum(x * math.log(x + 1e-12) for x in p if x > 0), 6)


def _minmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return [1.0 for _ in xs]
    return [(v - lo) / (hi - lo) for v in xs]


def extract_features(row: Dict[str, Any]) -> Dict[str, Any]:
    """15 observable features. Reads ONLY inference-safe candidate fields — no labels."""
    cands = row.get("candidates") or []
    fs = _first_stage_order(cands)
    rr = _reranker_order(cands)
    n = len(cands)
    fs_scores = [_fnum(c.get(_FS_SCORE)) for c in fs]
    rr_scores = [_fnum(c.get(_RR_SCORE)) for c in rr]

    def at(xs, i):
        return xs[i] if i < len(xs) else (xs[-1] if xs else 0.0)

    fs_ids = [c.get(_DID) for c in fs]
    rr_ids = [c.get(_DID) for c in rr]
    rr_pos = {d: i for i, d in enumerate(rr_ids)}
    fs_pos = {d: i for i, d in enumerate(fs_ids)}
    displacement = max((abs(rr_pos[d] - fs_pos[d]) for d in fs_ids if d in rr_pos), default=0)
    fs_top1_id = fs_ids[0] if fs_ids else None

    sources = [c.get(_SRC) for c in cands if c.get(_SRC)]
    nsrc = len(set(sources))
    src_agreement = (max((sources.count(s) for s in set(sources)), default=0) / len(sources)
                     if sources else 1.0)
    # bm25 vs dense agreement: overlap of their top-K ids if both sources present, else None
    bm25_ids = [c.get(_DID) for c in cands if c.get(_SRC) == "bm25"]
    dense_ids = [c.get(_DID) for c in cands if c.get(_SRC) in ("v3_dense", "dense", "e5_dense")]
    bm25_dense_agreement = (len(set(bm25_ids[:K]) & set(dense_ids[:K])) / K
                            if bm25_ids and dense_ids else None)

    return {
        "first_stage_top1_score": round(at(fs_scores, 0), 6),
        "first_stage_top2_score": round(at(fs_scores, 1), 6),
        "first_stage_top1_top2_gap": round(at(fs_scores, 0) - at(fs_scores, 1), 6),
        "first_stage_top1_top5_gap": round(at(fs_scores, 0) - at(fs_scores, 4), 6),
        "first_stage_score_entropy": _entropy(fs_scores),
        "candidate_source_agreement": round(src_agreement, 6),
        "bm25_dense_agreement": bm25_dense_agreement,
        "reranker_top1_score": round(at(rr_scores, 0), 6),
        "reranker_top2_score": round(at(rr_scores, 1), 6),
        "reranker_top1_top2_gap": round(at(rr_scores, 0) - at(rr_scores, 1), 6),
        "reranker_score_entropy": _entropy(rr_scores),
        "max_rank_displacement": displacement,
        "reranker_rank_of_first_stage_top1": rr_pos.get(fs_top1_id, 0),
        "num_candidates": n,
        "num_candidate_sources": nsrc,
    }


def _moves_top1_below(feat: Dict[str, Any], max_rank: int) -> bool:
    return feat["reranker_rank_of_first_stage_top1"] > max_rank


def apply_policy(row: Dict[str, Any], policy: str, params: Optional[Dict[str, Any]] = None
                 ) -> Tuple[List[str], str]:
    """Return (final_ranked_doc_ids, action). INFERENCE — uses features only, never labels."""
    params = params or {}
    cands = row.get("candidates") or []
    fs = _first_stage_order(cands)
    rr = _reranker_order(cands)
    fs_ids = [c.get(_DID) for c in fs]
    rr_ids = [c.get(_DID) for c in rr]
    feat = extract_features(row)

    def blend(alpha):
        fs_norm = {c.get(_DID): v for c, v in zip(fs, _minmax([_fnum(c.get(_FS_SCORE)) for c in fs]))}
        rr_norm = {c.get(_DID): v for c, v in zip(rr, _minmax([_fnum(c.get(_RR_SCORE)) for c in rr]))}
        scored = [(d, alpha * fs_norm.get(d, 0.0) + (1 - alpha) * rr_norm.get(d, 0.0)) for d in fs_ids]
        # stable: tie-break by first-stage order (index in fs_ids)
        order = sorted(range(len(scored)), key=lambda i: (-scored[i][1], i))
        return [scored[i][0] for i in order]

    fs_gap = params.get("fs_gap_threshold", 0.0)
    rr_gap = params.get("rr_gap_threshold", 0.0)
    alpha = params.get("alpha", 1.0)
    max_disp = params.get("max_displacement_rank", K)

    if policy == "always_rerank":
        return rr_ids, "rerank_top_k"
    if policy == "never_rerank":
        return fs_ids, "keep_first_stage"
    if policy == "first_stage_confidence_abstain":
        if feat["first_stage_top1_top2_gap"] >= fs_gap:
            return fs_ids, "keep_first_stage"
        return rr_ids, "rerank_top_k"
    if policy == "reranker_confidence_gate":
        if feat["reranker_top1_top2_gap"] >= rr_gap:
            return rr_ids, "rerank_top_k"
        return fs_ids, "keep_first_stage"
    if policy == "displacement_guard":
        if _moves_top1_below(feat, max_disp):
            return fs_ids, "keep_first_stage"
        return rr_ids, "rerank_top_k"
    if policy == "conservative_blend":
        return blend(alpha), "conservative_blend"
    if policy == "combined_policy":
        low_fs = feat["first_stage_top1_top2_gap"] < fs_gap          # first stage NOT confident
        high_rr = feat["reranker_top1_top2_gap"] >= rr_gap           # reranker confident
        if low_fs and high_rr and not _moves_top1_below(feat, max_disp):
            return rr_ids, "rerank_top_k"
        if low_fs and high_rr:               # wanted to rerank but displacement guard tripped
            return (blend(alpha), "conservative_blend") if alpha < 1.0 else (fs_ids, "keep_first_stage")
        return fs_ids, "keep_first_stage"
    raise ValueError(f"unknown policy: {policy}")


# ----------------------------------------------------------------- eval / fit (labels allowed)
def _positives(row: Dict[str, Any]) -> set:
    pos = set(row.get("positive_doc_ids") or [])
    if row.get("positive_doc_id"):
        pos.add(row["positive_doc_id"])
    pos |= {c.get(_DID) for c in (row.get("candidates") or []) if c.get("is_positive")}
    return pos


def _oracle_ndcg(row: Dict[str, Any], positives: set) -> float:
    ids = [c.get(_DID) for c in _first_stage_order(row.get("candidates") or [])]
    order = [d for d in ids if d in positives] + [d for d in ids if d not in positives]
    return ndcg_at_k(order, positives, K)


def evaluate_policy(rows: Sequence[Dict[str, Any]], policy: str, params: Dict[str, Any],
                    *, with_buckets: bool = True) -> Dict[str, Any]:
    """Apply the policy and score it (labels used HERE for analysis only, not at inference)."""
    from .hardness_aware_eval import assign_bucket
    fs_ndcgs, al_ndcgs, pol_ndcgs = [], [], []
    actions: Dict[str, int] = {}
    catastrophic = 0
    buckets: Dict[str, List[Tuple[float, float]]] = {}
    for r in rows:
        cands = r.get("candidates") or []
        if not cands:
            continue
        positives = _positives(r)
        fs_ids = [c.get(_DID) for c in _first_stage_order(cands)]
        rr_ids = [c.get(_DID) for c in _reranker_order(cands)]
        pol_ids, action = apply_policy(r, policy, params)
        fs_n = ndcg_at_k(fs_ids, positives, K)
        al_n = ndcg_at_k(rr_ids, positives, K)
        pol_n = ndcg_at_k(pol_ids, positives, K)
        fs_ndcgs.append(fs_n); al_ndcgs.append(al_n); pol_ndcgs.append(pol_n)
        actions[action] = actions.get(action, 0) + 1
        if pol_n - fs_n <= -CATASTROPHIC_DROP:
            catastrophic += 1
        if with_buckets:
            b = assign_bucket(fs_n, _oracle_ndcg(r, positives))
            buckets.setdefault(b, []).append((fs_n, pol_n))

    n = len(pol_ndcgs) or 1
    rerank_actions = sum(v for a, v in actions.items() if a != "keep_first_stage")

    def mean(xs):
        return round(sum(xs) / len(xs), 6) if xs else 0.0

    def bucket_delta(b):
        xs = buckets.get(b, [])
        return round(sum(p - f for f, p in xs) / len(xs), 6) if xs else None

    fs_mean, pol_mean, al_mean = mean(fs_ndcgs), mean(pol_ndcgs), mean(al_ndcgs)
    medium_hard = buckets.get("medium", []) + buckets.get("hard", [])
    return {
        "policy": policy, "params": params, "n_queries": len(pol_ndcgs),
        "first_stage_ndcg@10": fs_mean, "always_rerank_ndcg@10": al_mean,
        "policy_ndcg@10": pol_mean,
        "delta_vs_first_stage": round(pol_mean - fs_mean, 6),
        "delta_vs_always_rerank": round(pol_mean - al_mean, 6),
        "abstain_rate": round(actions.get("keep_first_stage", 0) / n, 6),
        "rerank_rate": round(rerank_actions / n, 6),
        "catastrophic_drop_rate": round(catastrophic / n, 6),
        "actions": actions,
        "medium_hard_delta": (round(sum(p - f for f, p in medium_hard) / len(medium_hard), 6)
                              if medium_hard else None),
        "by_bucket": {b: {"n": len(buckets.get(b, [])),
                          "first_stage_ndcg@10": round(sum(f for f, _ in buckets[b]) / len(buckets[b]), 6),
                          "policy_ndcg@10": round(sum(p for _, p in buckets[b]) / len(buckets[b]), 6),
                          "delta": bucket_delta(b)}
                      for b in sorted(buckets)},
    }


def grid_search(dev_rows: Sequence[Dict[str, Any]], *, policy: str = "combined_policy",
                fs_gaps: Sequence[float], rr_gaps: Sequence[float],
                alphas: Sequence[float] = (1.0,),
                max_displacements: Sequence[int] = (K,),
                abstain_target: Optional[float] = None) -> Dict[str, Any]:
    """Fit thresholds on DEV ONLY (dev labels allowed). Maximize dev policy nDCG@10; tie-break to
    higher abstain (more conservative) then lower catastrophic. Guardrail sets are never passed in."""
    best = None
    trials = []
    for fg in fs_gaps:
        for rg in rr_gaps:
            for al in alphas:
                for md in max_displacements:
                    params = {"fs_gap_threshold": fg, "rr_gap_threshold": rg, "alpha": al,
                              "max_displacement_rank": md}
                    rep = evaluate_policy(dev_rows, policy, params, with_buckets=False)
                    if abstain_target is not None and rep["abstain_rate"] < abstain_target:
                        # still allow, but record; selection prefers meeting target via tie-break
                        pass
                    key = (rep["policy_ndcg@10"], rep["abstain_rate"], -rep["catastrophic_drop_rate"])
                    trials.append({"params": params, "dev_ndcg": rep["policy_ndcg@10"],
                                   "dev_delta": rep["delta_vs_first_stage"],
                                   "abstain_rate": rep["abstain_rate"],
                                   "catastrophic": rep["catastrophic_drop_rate"]})
                    if best is None or key > best["key"]:
                        best = {"key": key, "params": params, "dev": rep}
    return {"policy": policy, "best_params": best["params"], "dev_metrics": best["dev"],
            "fit_on": "dev_only", "n_trials": len(trials), "trials": trials[:200]}


POLICY_GATE = {
    "webfaq_min_overall": 0.05, "webfaq_min_medium_hard": 0.20,
    "germanquad_min_overall": -0.005, "germanquad_max_catastrophic": 0.03,
    "dt_test_min_overall": -0.005, "dt_test_max_catastrophic": 0.02,
}


def policy_gate(reports: Dict[str, Dict[str, Any]],
                always_rerank: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Promotion gate for the policy. ``reports``/``always_rerank``: set name -> evaluate_policy()."""
    checks = []

    def chk(name, ok, detail):
        checks.append({"check": name, "status": "pass" if ok else "fail", "detail": detail})

    wf = reports.get("webfaq", {})
    if wf:
        chk("webfaq_overall", wf["delta_vs_first_stage"] >= POLICY_GATE["webfaq_min_overall"] - 1e-9,
            f"{wf['delta_vs_first_stage']:+.4f} (min +{POLICY_GATE['webfaq_min_overall']})")
        mh = wf.get("medium_hard_delta")
        chk("webfaq_medium_hard", mh is not None and mh >= POLICY_GATE["webfaq_min_medium_hard"] - 1e-9,
            f"{mh} (min +{POLICY_GATE['webfaq_min_medium_hard']})")
    for g, mino, maxc in (("germanquad", POLICY_GATE["germanquad_min_overall"], POLICY_GATE["germanquad_max_catastrophic"]),
                          ("dt_test", POLICY_GATE["dt_test_min_overall"], POLICY_GATE["dt_test_max_catastrophic"])):
        r = reports.get(g)
        if r:
            chk(f"{g}_overall", r["delta_vs_first_stage"] >= mino - 1e-9,
                f"{r['delta_vs_first_stage']:+.4f} (min {mino})")
            chk(f"{g}_catastrophic", r["catastrophic_drop_rate"] <= maxc + 1e-9,
                f"{r['catastrophic_drop_rate']:.4f} (max {maxc})")
            ar = always_rerank.get(g)
            if ar:
                chk(f"{g}_beats_always_rerank",
                    r["delta_vs_first_stage"] > ar["delta_vs_first_stage"] - 1e-9,
                    f"policy {r['delta_vs_first_stage']:+.4f} vs always_rerank {ar['delta_vs_first_stage']:+.4f}")
    failing = [c for c in checks if c["status"] == "fail"]
    return {"status": "pass" if not failing else "fail", "checks": checks, "failing": failing,
            "thresholds": POLICY_GATE}
