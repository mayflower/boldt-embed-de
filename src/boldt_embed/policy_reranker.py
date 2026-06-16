"""Production-style serving wrapper that applies a bounded rerank policy to already-scored
candidates (pure stdlib, no ML). Raw always-rerank is IMPOSSIBLE unless explicitly requested with
``allow_raw=True`` (CLI: ``--allow-raw-rerank-dangerous``).

The policy decision uses ONLY observable per-candidate fields (first_stage_rank/score,
reranker_score, candidate_source, doc_id) — never qrels/labels/oracle/hardness/eval-set. Output is
the per-candidate final ranking with a ``policy_action`` + ``policy_reason`` and per-query
diagnostics. Deterministic tie-breaks throughout.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .policy_config import load_policy
from .rerank_abstain import _fnum, _minmax

MODES = ("policy_gated", "first_stage_only", "raw_rerank")
_FS_RANK, _FS_SCORE, _RR, _SRC, _DID = ("first_stage_rank", "first_stage_score", "reranker_score",
                                        "candidate_source", "doc_id")


def _validate_row(row: Any) -> None:
    if not isinstance(row, dict):
        raise ValueError("candidate list row must be a JSON object")
    if not row.get("query_id"):
        raise ValueError("candidate list row missing 'query_id'")
    cands = row.get("candidates")
    if not isinstance(cands, list) or not cands:
        raise ValueError(f"query {row.get('query_id')}: 'candidates' must be a non-empty list")
    for i, c in enumerate(cands):
        if not isinstance(c, dict) or not c.get(_DID):
            raise ValueError(f"query {row.get('query_id')}: candidate[{i}] missing 'doc_id'")
        if c.get(_FS_RANK) is None and c.get(_FS_SCORE) is None:
            raise ValueError(f"query {row.get('query_id')}: candidate[{i}] needs first_stage_rank "
                             "or first_stage_score")


def _orders(cands):
    if all(c.get(_FS_RANK) is not None for c in cands):
        fs = sorted(cands, key=lambda c: (_fnum(c[_FS_RANK], 1e9), str(c.get(_DID))))
    else:
        fs = sorted(cands, key=lambda c: (-_fnum(c.get(_FS_SCORE)), str(c.get(_DID))))
    fs_ids = [c[_DID] for c in fs]
    has_rr = all(c.get(_RR) is not None for c in cands)
    rr_ids = ([c[_DID] for c in sorted(cands, key=lambda c: (-_fnum(c.get(_RR)), str(c.get(_DID))))]
              if has_rr else list(fs_ids))
    return fs_ids, rr_ids, has_rr


def _bounded_tail(tail_ids, fs_rank, score, *, max_down, max_up, exempt, start_pos):
    """Order the non-locked tail into absolute positions start_pos.. respecting max_downshift
    (hard deadline) and max_upshift_without_margin (release; exempt docs ignore it). Deterministic."""
    remaining = list(tail_ids)
    placed: List[str] = []
    for p in range(start_pos, start_pos + len(tail_ids)):
        overdue = [d for d in remaining if fs_rank[d] + max_down <= p]
        if overdue:
            pick = min(overdue, key=lambda d: (fs_rank[d] + max_down, -score[d], str(d)))
        else:
            elig = [d for d in remaining if (fs_rank[d] - max_up) <= p or d in exempt]
            pool = elig or remaining
            pick = max(pool, key=lambda d: (score[d], -fs_rank[d], _negkey(d)))
        placed.append(pick)
        remaining.remove(pick)
    return placed


def _negkey(doc_id: str):
    return tuple(-ord(ch) for ch in str(doc_id))   # deterministic, order-independent


def rerank_query(row: Dict[str, Any], policy: Dict[str, Any], *, mode: str = "policy_gated",
                 allow_raw: bool = False) -> Dict[str, Any]:
    _validate_row(row)
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if mode == "raw_rerank" and not allow_raw:
        raise ValueError("raw_rerank is unsafe and disabled by default; pass allow_raw=True "
                         "(--allow-raw-rerank-dangerous) to override")
    cands = row["candidates"]
    by_id = {c[_DID]: c for c in cands}
    fs_ids, rr_ids, has_rr = _orders(cands)
    fs_rank = {d: i for i, d in enumerate(fs_ids)}
    rr_rank = {d: i for i, d in enumerate(rr_ids)}
    rr_score = {d: _fnum(by_id[d].get(_RR)) for d in fs_ids}
    fs_score = {d: _fnum(by_id[d].get(_FS_SCORE)) for d in fs_ids}
    b = policy.get("bounds", {})
    preserve_k = int(b.get("preserve_first_stage_top_k", 3))
    max_down = int(b.get("max_downshift", 2))
    max_up = int(b.get("max_upshift_without_margin", 5))
    margin = float(b.get("margin_override", 3.0))
    alpha_hi = float(b.get("blend_alpha_high_confidence", 0.85))
    alpha_def = float(b.get("blend_alpha_default", 0.65))

    actions: Dict[str, str] = {}
    reasons: Dict[str, str] = {}
    margin_override_used = False
    final_score: Dict[str, float] = {}

    if mode == "first_stage_only" or not has_rr:
        final = list(fs_ids)
        for d in final:
            actions[d] = "kept_first_stage"
            reasons[d] = ("first_stage_only mode" if mode == "first_stage_only"
                          else "no reranker_score present; kept first stage")
    elif mode == "raw_rerank":
        final = list(rr_ids)
        for d in final:
            actions[d] = "reranked"
            reasons[d] = "raw always-rerank (DANGEROUS mode explicitly enabled)"
    else:  # policy_gated — bounded margin_override
        fs_top1 = fs_ids[0]
        override_doc = max(fs_ids, key=lambda d: (rr_score[d], -fs_rank[d]))
        margin_override_used = (rr_score[override_doc] - rr_score[fs_top1]) >= margin and \
            override_doc != fs_top1
        alpha = alpha_def if margin_override_used else alpha_hi
        # locked head
        if margin_override_used:
            head = [override_doc] + [d for d in fs_ids[:preserve_k] if d != override_doc]
            head = head[:preserve_k]
        else:
            head = list(fs_ids[:preserve_k])
        head_set = set(head)
        for d in head:
            if d == override_doc and margin_override_used:
                actions[d] = "margin_override"
                reasons[d] = (f"reranker beats first-stage top1 by >= {margin}; promoted to rank 1")
            else:
                actions[d] = "locked"
                reasons[d] = f"first-stage top-{preserve_k} preserved"
        # blended tail, bounded
        tail = [d for d in fs_ids if d not in head_set]
        fsn = dict(zip(fs_ids, _minmax([fs_score[d] for d in fs_ids])))
        rrn = dict(zip(fs_ids, _minmax([rr_score[d] for d in fs_ids])))
        blended = {d: alpha * fsn[d] + (1 - alpha) * rrn[d] for d in fs_ids}
        exempt = {d for d in tail if (rr_score[d] - rr_score[fs_top1]) >= margin}
        tail_order = _bounded_tail(tail, fs_rank, blended, max_down=max_down, max_up=max_up,
                                   exempt=exempt, start_pos=len(head))
        final = head + tail_order
        for d in tail_order:
            final_score[d] = round(blended[d], 6)
            if final.index(d) == fs_rank[d]:
                actions[d] = "kept_first_stage"
                reasons[d] = "blended order matched first stage"
            else:
                actions[d] = "blended"
                reasons[d] = f"blended (alpha={alpha}) within downshift<={max_down}/upshift<={max_up}"
        for d in head:
            final_score[d] = round(blended[d], 6)

    out_c = []
    for pos, d in enumerate(final):
        rec = {"doc_id": d, "final_rank": pos, "first_stage_rank": fs_rank[d],
               "policy_action": actions[d], "policy_reason": reasons[d]}
        if has_rr:
            rec["reranker_rank"] = rr_rank[d]
        if d in final_score:
            rec["final_score"] = final_score[d]
        out_c.append(rec)

    downs = [final.index(d) - fs_rank[d] for d in fs_ids]
    diagnostics = {
        "max_downshift": max([x for x in downs] + [0]),
        "max_upshift": max([-x for x in downs] + [0]),
        "top_k_locked": (sum(1 for d in final[:preserve_k] if actions.get(d) == "locked")
                         if mode == "policy_gated" else 0),
        "margin_override_used": margin_override_used,
        "num_candidates": len(cands),
    }
    return {"query_id": row["query_id"], "policy_id": policy.get("policy_id"),
            "ranking_mode": mode if mode != "policy_gated" else "policy_gated",
            "candidates": out_c, "diagnostics": diagnostics}


def rerank_lists(rows, policy, *, mode="policy_gated", allow_raw=False):
    return [rerank_query(r, policy, mode=mode, allow_raw=allow_raw) for r in rows]
