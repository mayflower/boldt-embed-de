"""Inference-only bounded reranking policies over already-scored candidate lists (pure stdlib).

Wraps the EXISTING (conservative) v5 reranker scores with bounded policies that cap how far the
reranker may disturb a confident first stage — directly targeting the residual catastrophic rank
churn on near-ceiling GermanQuAD lists. No training, no Qwen, no rescoring.

INFERENCE USES OBSERVABLE FEATURES ONLY (via rerank_abstain.extract_features) +
first_stage_rank / reranker_score / doc_id. It NEVER reads qrels / labels / oracle /
hardness_bucket / eval-set name. Those are used only in fit (dev labels) and eval/analysis.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .metrics import ndcg_at_k
from .rerank_abstain import _fnum, _minmax, extract_features

K = 10
CATASTROPHIC_DROP = 0.2

POLICIES = ("identity", "always_rerank", "top1_lock", "topk_lock", "bounded_downshift",
            "bounded_upshift", "margin_override", "blend", "confidence_conditional",
            "combined_safe_policy")
_FS_RANK, _FS_SCORE, _RR, _DID = ("first_stage_rank", "first_stage_score", "reranker_score", "doc_id")


def _first_stage(cands):
    if cands and all(c.get(_FS_RANK) is not None for c in cands):
        return sorted(cands, key=lambda c: _fnum(c[_FS_RANK], 1e9))
    if cands and all(c.get(_FS_SCORE) is not None for c in cands):
        return sorted(cands, key=lambda c: -_fnum(c[_FS_SCORE]))
    return list(cands)


def _reranker(cands):
    return sorted(cands, key=lambda c: (-_fnum(c.get(_RR)), str(c.get(_DID))))


def _ctx(row):
    cands = row.get("candidates") or []
    fs = _first_stage(cands)
    fs_ids = [c.get(_DID) for c in fs]
    rr_ids = [c.get(_DID) for c in _reranker(cands)]
    fs_rank = {d: i for i, d in enumerate(fs_ids)}
    rr = {c.get(_DID): _fnum(c.get(_RR)) for c in cands}
    fs_score = {c.get(_DID): _fnum(c.get(_FS_SCORE)) for c in cands}
    return fs_ids, rr_ids, fs_rank, rr, fs_score


def _bounded_downshift(fs_ids, fs_rank, rr, D) -> List[str]:
    """No doc moves down more than D ranks (final_rank <= fs_rank + D). Earliest-deadline-first
    when forced, else best-reranker; deterministic tie-breaks."""
    remaining = list(fs_ids)
    placed: List[str] = []
    for p in range(len(fs_ids)):
        overdue = [d for d in remaining if fs_rank[d] + D <= p]
        if overdue:
            pick = min(overdue, key=lambda d: (fs_rank[d] + D, -rr[d], str(d)))
        else:
            pick = max(remaining, key=lambda d: (rr[d], -fs_rank[d], _neg(d)))
        placed.append(pick)
        remaining.remove(pick)
    return placed


def _bounded_upshift(fs_ids, fs_rank, rr, U, margin, fs_top1) -> List[str]:
    """No doc moves up more than U ranks (final_rank >= fs_rank - U) unless its reranker score
    beats the first-stage top1 by >= margin."""
    exempt = {d for d in fs_ids if rr[d] - rr[fs_top1] >= margin}
    remaining = list(fs_ids)
    placed: List[str] = []
    for p in range(len(fs_ids)):
        eligible = [d for d in remaining if (fs_rank[d] - U) <= p or d in exempt]
        pick = max(eligible, key=lambda d: (rr[d], -fs_rank[d], _neg(d)))
        placed.append(pick)
        remaining.remove(pick)
    return placed


def _neg(doc_id: str) -> int:
    # deterministic, order-independent tie-break that prefers a stable lexicographic order
    return -sum((i + 1) * ord(ch) for i, ch in enumerate(str(doc_id)))


def _blend(fs_ids, fs_rank, rr, fs_score, alpha) -> List[str]:
    fsn = dict(zip(fs_ids, _minmax([fs_score[d] for d in fs_ids])))
    rrn = dict(zip(fs_ids, _minmax([rr[d] for d in fs_ids])))
    scored = [(d, alpha * fsn[d] + (1 - alpha) * rrn[d]) for d in fs_ids]
    order = sorted(range(len(scored)), key=lambda i: (-scored[i][1], i))  # tie -> first-stage order
    return [scored[i][0] for i in order]


def apply_policy(row: Dict[str, Any], policy: str, params: Optional[Dict[str, Any]] = None
                 ) -> Tuple[List[str], str]:
    """Return (final_ranked_doc_ids, action). INFERENCE — observable features only, no labels."""
    params = params or {}
    fs_ids, rr_ids, fs_rank, rr, fs_score = _ctx(row)
    if not fs_ids:
        return [], "identity"
    feat = extract_features(row)
    k = int(params.get("k", 3))
    D = int(params.get("D", 3))
    U = int(params.get("U", 3))
    alpha = float(params.get("alpha", 1.0))
    margin = float(params.get("margin", 2.0))
    fs_gap_high = float(params.get("fs_gap_high", params.get("fs_gap_threshold", 1e9)))
    fs_gap_med = float(params.get("fs_gap_med", 0.0))
    fs_top1 = fs_ids[0]

    if policy == "identity":
        return fs_ids, "identity"
    if policy == "always_rerank":
        return rr_ids, "rerank"
    if policy == "top1_lock":
        return [fs_top1] + [d for d in rr_ids if d != fs_top1], "top1_lock"
    if policy == "topk_lock":
        head = fs_ids[:k]
        hs = set(head)
        return head + [d for d in rr_ids if d not in hs], "topk_lock"
    if policy == "bounded_downshift":
        return _bounded_downshift(fs_ids, fs_rank, rr, D), "bounded_downshift"
    if policy == "bounded_upshift":
        return _bounded_upshift(fs_ids, fs_rank, rr, U, margin, fs_top1), "bounded_upshift"
    if policy == "margin_override":
        rr_top1 = max(rr.values()) if rr else 0.0
        if rr_top1 - rr[fs_top1] >= margin:
            return rr_ids, "override_rerank"
        return [fs_top1] + [d for d in rr_ids if d != fs_top1], "top1_lock"
    if policy == "blend":
        return _blend(fs_ids, fs_rank, rr, fs_score, alpha), "blend"
    if policy == "confidence_conditional":
        if feat["first_stage_top1_top2_gap"] >= fs_gap_high:
            head = fs_ids[:k]; hs = set(head)
            return head + [d for d in rr_ids if d not in hs], "topk_lock"
        return rr_ids, "rerank"
    if policy == "combined_safe_policy":
        gap = feat["first_stage_top1_top2_gap"]
        if gap >= fs_gap_high:                         # high confidence -> top-k lock + blend
            head = fs_ids[:k]; hs = set(head)
            tail = _blend([d for d in fs_ids if d not in hs], fs_rank, rr, fs_score, alpha)
            return head + tail, "high_topk_lock_blend"
        if gap >= fs_gap_med:                           # medium -> bounded downshift + margin override
            rr_top1 = max(rr.values()) if rr else 0.0
            if rr_top1 - rr[fs_top1] >= margin:
                return _bounded_downshift(fs_ids, fs_rank, rr, D), "med_downshift_override"
            return _bounded_downshift([fs_top1] + [d for d in fs_ids if d != fs_top1],
                                      fs_rank, rr, D), "med_downshift"
        return _bounded_upshift(fs_ids, fs_rank, rr, U, margin, fs_top1), "low_bounded_upshift"
    raise ValueError(f"unknown policy: {policy}")


# ----------------------------------------------------------------- eval / fit (labels allowed)
def _positives(row):
    pos = set(row.get("positive_doc_ids") or [])
    if row.get("positive_doc_id"):
        pos.add(row["positive_doc_id"])
    pos |= {c.get(_DID) for c in (row.get("candidates") or []) if c.get("is_positive")}
    return pos


def _oracle(row, positives):
    ids = [c.get(_DID) for c in _first_stage(row.get("candidates") or [])]
    return ndcg_at_k([d for d in ids if d in positives] + [d for d in ids if d not in positives],
                     positives, K)


def evaluate_policy(rows: Sequence[Dict[str, Any]], policy: str, params: Dict[str, Any],
                    *, with_buckets: bool = True, top_catastrophic: int = 10) -> Dict[str, Any]:
    from .hardness_aware_eval import assign_bucket
    fs_n, al_n, pol_n = [], [], []
    actions: Dict[str, int] = {}
    cat = 0
    disp: List[int] = []
    locks = 0
    abst = 0
    buckets: Dict[str, List[Tuple[float, float]]] = {}
    worst: List[Dict[str, Any]] = []
    for r in rows:
        cands = r.get("candidates") or []
        if not cands:
            continue
        positives = _positives(r)
        fs_ids = [c.get(_DID) for c in _first_stage(cands)]
        rr_ids = [c.get(_DID) for c in _reranker(cands)]
        pol_ids, action = apply_policy(r, policy, params)
        actions[action] = actions.get(action, 0) + 1
        f = ndcg_at_k(fs_ids, positives, K)
        a = ndcg_at_k(rr_ids, positives, K)
        p = ndcg_at_k(pol_ids, positives, K)
        fs_n.append(f); al_n.append(a); pol_n.append(p)
        fs_rank = {d: i for i, d in enumerate(fs_ids)}
        disp.append(max((abs(pol_ids.index(d) - fs_rank[d]) for d in fs_ids), default=0))
        if pol_ids[:1] == fs_ids[:1]:
            locks += 1
        if pol_ids == fs_ids:
            abst += 1
        if p - f <= -CATASTROPHIC_DROP:
            cat += 1
            worst.append({"query_id": r.get("query_id"), "first_stage_ndcg@10": round(f, 4),
                          "policy_ndcg@10": round(p, 4), "delta": round(p - f, 4)})
        if with_buckets:
            buckets.setdefault(assign_bucket(f, _oracle(r, positives)), []).append((f, p))

    n = len(pol_n) or 1

    def mean(xs):
        return round(sum(xs) / len(xs), 6) if xs else 0.0

    medium_hard = buckets.get("medium", []) + buckets.get("hard", [])
    worst.sort(key=lambda x: x["delta"])
    return {
        "policy": policy, "params": params, "n_queries": len(pol_n),
        "first_stage_ndcg@10": mean(fs_n), "always_rerank_ndcg@10": mean(al_n),
        "policy_ndcg@10": mean(pol_n),
        "delta_vs_first_stage": round(mean(pol_n) - mean(fs_n), 6),
        "delta_vs_always_rerank": round(mean(pol_n) - mean(al_n), 6),
        "abstain_rate": round(abst / n, 6), "lock_rate": round(locks / n, 6),
        "avg_max_displacement": round(sum(disp) / n, 4),
        "catastrophic_drop_rate": round(cat / n, 6),
        "medium_hard_delta": (round(sum(p - f for f, p in medium_hard) / len(medium_hard), 6)
                              if medium_hard else None),
        "actions": actions,
        "by_bucket": {b: {"n": len(buckets[b]),
                          "first_stage_ndcg@10": round(sum(f for f, _ in buckets[b]) / len(buckets[b]), 6),
                          "policy_ndcg@10": round(sum(p for _, p in buckets[b]) / len(buckets[b]), 6),
                          "delta": round(sum(p - f for f, p in buckets[b]) / len(buckets[b]), 6)}
                      for b in sorted(buckets)},
        "top_catastrophic_examples": worst[:top_catastrophic],
    }


def candidate_grid(dev_rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """Policy/param candidates for the dev grid search (data-adaptive fs-gap thresholds)."""
    feats = [extract_features(r) for r in dev_rows]
    gaps = sorted(f["first_stage_top1_top2_gap"] for f in feats)

    def q(p):
        return gaps[min(len(gaps) - 1, max(0, int(round(p * (len(gaps) - 1)))))] if gaps else 0.0
    grid: List[Tuple[str, Dict[str, Any]]] = [("identity", {}), ("always_rerank", {}),
                                              ("top1_lock", {})]
    for k in (1, 2, 3, 5, 10):
        grid.append(("topk_lock", {"k": k}))
    for D in (1, 2, 3, 5, 10):
        grid.append(("bounded_downshift", {"D": D}))
    for U in (1, 2, 3, 5):
        for M in (1.0, 2.0, 5.0):
            grid.append(("bounded_upshift", {"U": U, "margin": M}))
    for M in (1.0, 2.0, 3.0, 5.0):
        grid.append(("margin_override", {"margin": M}))
    for a in (0.1, 0.3, 0.5, 0.7, 0.9):
        grid.append(("blend", {"alpha": a}))
    for thr in (q(0.5), q(0.75)):
        for k in (1, 3):
            grid.append(("confidence_conditional", {"fs_gap_high": thr, "k": k}))
    for k in (1, 3):
        for D in (2, 3):
            grid.append(("combined_safe_policy",
                         {"fs_gap_high": q(0.75), "fs_gap_med": q(0.4), "k": k, "D": D,
                          "margin": 2.0, "U": 3, "alpha": 0.5}))
    return grid


def grid_search(dev_rows: Sequence[Dict[str, Any]], *, catastrophic_target: float = 0.03,
                hc_gap_percentile: float = 0.6, safety_top1_keep_min: float = 0.9
                ) -> Dict[str, Any]:
    """Fit on DEV ONLY. Deployment-safety is enforced with an OBSERVABLE constraint (no guardrail
    labels): on high-first-stage-confidence dev lists (gap >= percentile), the policy must keep the
    confident first-stage top-1 at rank 1 on >= ``safety_top1_keep_min`` of them. Among such "safe"
    policies, pick the highest dev nDCG@10. This selects a protective policy even when the dev set
    itself does not exhibit catastrophic churn (the WebFAQ blind spot), and it transfers to the
    guardrails via the same observable feature."""
    feats = [extract_features(r) for r in dev_rows]
    gaps = sorted(f["first_stage_top1_top2_gap"] for f in feats)
    gap_thr = gaps[min(len(gaps) - 1, max(0, int(round(hc_gap_percentile * (len(gaps) - 1)))))] \
        if gaps else 0.0
    hc_rows = [r for r, f in zip(dev_rows, feats) if f["first_stage_top1_top2_gap"] >= gap_thr]

    def _top1_keep(policy, params):
        if not hc_rows:
            return 1.0
        kept = 0
        for r in hc_rows:
            fs_top1 = [c.get(_DID) for c in _first_stage(r.get("candidates") or [])][:1]
            final, _ = apply_policy(r, policy, params)   # apply_policy returns (doc_ids, action)
            if final[:1] == fs_top1:
                kept += 1
        return kept / len(hc_rows)

    trials = []
    for policy, params in candidate_grid(dev_rows):
        rep = evaluate_policy(dev_rows, policy, params, with_buckets=False)
        trials.append({"policy": policy, "params": params, "dev_ndcg": rep["policy_ndcg@10"],
                       "dev_delta": rep["delta_vs_first_stage"],
                       "catastrophic": rep["catastrophic_drop_rate"],
                       "medium_hard_delta": rep["medium_hard_delta"],
                       "hc_top1_keep": round(_top1_keep(policy, params), 4)})
    safe = [t for t in trials if t["hc_top1_keep"] >= safety_top1_keep_min - 1e-9]
    pool = safe or trials
    best = max(pool, key=lambda t: (t["dev_ndcg"], t["hc_top1_keep"], -t["catastrophic"]))
    return {"policy": best["policy"], "best_params": best["params"],
            "fit_on": "dev_only", "catastrophic_target": catastrophic_target,
            "safety": {"hc_gap_threshold": round(gap_thr, 4), "hc_lists": len(hc_rows),
                       "safety_top1_keep_min": safety_top1_keep_min,
                       "selected_safe": bool(safe)},
            "dev_metrics": best, "n_trials": len(trials),
            "trials": sorted(trials, key=lambda t: (-t["hc_top1_keep"], -t["dev_ndcg"]))[:60]}


BOUNDED_GATE = {
    "germanquad_min_overall": -0.005, "germanquad_max_catastrophic": 0.03,
    "dt_test_min_overall": -0.005, "dt_test_max_catastrophic": 0.02,
    "webfaq_min_overall": 0.05,
}


def bounded_policy_gate(reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Task gate target (7 checks). ``reports``: set name -> evaluate_policy() (carries
    always_rerank_delta_vs_first_stage). No dt_test-beats-always-rerank check here."""
    checks = []

    def chk(name, ok, detail):
        checks.append({"check": name, "status": "pass" if ok else "fail", "detail": detail})

    wf, gq, dt = reports.get("webfaq", {}), reports.get("germanquad", {}), reports.get("dt_test", {})
    if wf:
        chk("webfaq_overall", wf["delta_vs_first_stage"] >= BOUNDED_GATE["webfaq_min_overall"] - 1e-9,
            f"{wf['delta_vs_first_stage']:+.4f} (min +{BOUNDED_GATE['webfaq_min_overall']})")
        mh = wf.get("medium_hard_delta")
        chk("webfaq_medium_hard_positive", mh is not None and mh > 0,
            f"{mh} (must be > 0)")
    if gq:
        chk("germanquad_overall", gq["delta_vs_first_stage"] >= BOUNDED_GATE["germanquad_min_overall"] - 1e-9,
            f"{gq['delta_vs_first_stage']:+.4f} (min {BOUNDED_GATE['germanquad_min_overall']})")
        chk("germanquad_catastrophic", gq["catastrophic_drop_rate"] <= BOUNDED_GATE["germanquad_max_catastrophic"] + 1e-9,
            f"{gq['catastrophic_drop_rate']:.4f} (max {BOUNDED_GATE['germanquad_max_catastrophic']})")
        ar = gq.get("always_rerank_delta_vs_first_stage")
        chk("germanquad_beats_raw_always_rerank", ar is not None and gq["delta_vs_first_stage"] > ar - 1e-9,
            f"policy {gq['delta_vs_first_stage']:+.4f} vs always_rerank {ar}")
    if dt:
        chk("dt_test_overall", dt["delta_vs_first_stage"] >= BOUNDED_GATE["dt_test_min_overall"] - 1e-9,
            f"{dt['delta_vs_first_stage']:+.4f} (min {BOUNDED_GATE['dt_test_min_overall']})")
        chk("dt_test_catastrophic", dt["catastrophic_drop_rate"] <= BOUNDED_GATE["dt_test_max_catastrophic"] + 1e-9,
            f"{dt['catastrophic_drop_rate']:.4f} (max {BOUNDED_GATE['dt_test_max_catastrophic']})")
    failing = [c for c in checks if c["status"] == "fail"]
    return {"status": "pass" if not failing else "fail", "checks": checks, "failing": failing,
            "thresholds": BOUNDED_GATE}
