"""Rank-preservation loss for the conservative v5 reranker.

v5 always-rerank helps medium+hard lists but churns near-ceiling GermanQuAD. This adds a training
penalty that discourages reordering a HIGH-FIRST-STAGE-CONFIDENCE list unless the teacher margin
strongly justifies the move. Total loss (see `scripts/train_v5_rag_reranker_conservative.py`):

    listwise_teacher_kl + pairwise_margin + pointwise_confident_bce + lambda_preserve * preservation

The penalty uses ONLY teacher scores + first-stage ranks (training-time signals); it never uses eval
qrels. High-confidence detection uses ONLY observable features (first-stage gap/entropy/source
agreement) — the same features the abstention policy uses. Module import is torch-free; the tensor
loss imports torch lazily inside the function.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .rerank_abstain import extract_features

DEFAULT_JUSTIFY_MARGIN = 2.0          # teacher advantage needed to "justify" an inversion
DEFAULT_FS_GAP_PERCENTILE = 0.6       # lists with first-stage gap >= this percentile are "confident"


def is_high_confidence(row: Dict[str, Any], *, fs_gap_min: float,
                       entropy_max: Optional[float] = None,
                       source_agreement_min: float = 1.0) -> bool:
    """High first-stage confidence from OBSERVABLE features only (no qrels/labels)."""
    f = extract_features(row)
    if f["first_stage_top1_top2_gap"] < fs_gap_min:
        return False
    if entropy_max is not None and f["first_stage_score_entropy"] > entropy_max:
        return False
    if f["candidate_source_agreement"] < source_agreement_min:
        return False
    return True


def plan_conservative_loss(lambda_preserve: float) -> Dict[str, Any]:
    """Describe the conservative loss stack (stdlib, no ML) for the run card / dry-run."""
    return {
        "components": ["KLDivLoss(listwise)", "MarginRankingLoss",
                       "BCEWithLogitsLoss(high_confidence_only)",
                       f"rank_preservation_loss(lambda={lambda_preserve})"],
        "weights": {"listwise": 1.0, "pairwise": 0.5, "pointwise_bce": 0.2,
                    "preservation": lambda_preserve},
        "primary": "listwise",
        "preservation_applies_to": "high_first_stage_confidence_lists_only",
        "uncertain_listwise_only": True,
    }


def _percentile(values: Sequence[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def _softmax(scores: List[float]) -> List[float]:
    import math
    if not scores:
        return []
    m = max(scores)
    e = [math.exp(s - m) for s in scores]
    tot = sum(e) or 1.0
    return [x / tot for x in e]


def scored_lists_to_conservative_batches(rows: Sequence[Dict[str, Any]], *,
                                         fs_gap_min: Optional[float] = None,
                                         fs_gap_percentile: float = DEFAULT_FS_GAP_PERCENTILE,
                                         entropy_max: Optional[float] = None
                                         ) -> List[Dict[str, Any]]:
    """Build listwise batches that also carry first_stage_ranks, teacher_scores, and a
    high_confidence flag (so the trainer can apply the preservation penalty selectively). Stdlib."""
    feats = [extract_features(r) for r in rows]
    if fs_gap_min is None:
        fs_gap_min = _percentile([f["first_stage_top1_top2_gap"] for f in feats], fs_gap_percentile)
    out: List[Dict[str, Any]] = []
    for r, f in zip(rows, feats):
        cands = r.get("candidates") or []
        if len(cands) < 2:
            continue
        docs = [c.get("document") or c.get("text", "") for c in cands]
        tscores = [float(c.get("teacher_score")) if c.get("teacher_score") is not None else -10.0
                   for c in cands]
        tgt = [c.get("teacher_softmax_target") for c in cands]
        target = ([float(t) for t in tgt] if all(t is not None for t in tgt)
                  and abs(sum(tgt) - 1.0) < 1e-3 else _softmax(tscores))
        ranks = [int(c["first_stage_rank"]) if c.get("first_stage_rank") is not None else i
                 for i, c in enumerate(cands)]
        out.append({"query": r.get("query", ""), "documents": docs, "target": target,
                    "teacher_scores": tscores, "first_stage_ranks": ranks,
                    "high_confidence": is_high_confidence(r, fs_gap_min=fs_gap_min,
                                                          entropy_max=entropy_max)})
    return out


def rank_preservation_loss(student_scores, first_stage_ranks: Sequence[int], teacher_scores,
                           *, justify_margin: float = DEFAULT_JUSTIFY_MARGIN):
    """Penalize student inversions of the first-stage order that the teacher does NOT justify.

    For every first-stage-ordered pair (i higher than j), if the teacher does not advantage j over
    i by >= ``justify_margin``, penalize the student for scoring j above i: relu(s_j - s_i). Returns
    0 when the student preserves the first-stage order, and is large when it moves docs (incl. the
    first-stage top1) above better-first-stage docs without teacher support. Differentiable wrt
    ``student_scores``; teacher_scores / first_stage_ranks are constants. (torch, lazy import.)"""
    import torch
    ss = student_scores if torch.is_tensor(student_scores) else torch.as_tensor(
        student_scores, dtype=torch.float32)
    ts = teacher_scores if torch.is_tensor(teacher_scores) else torch.as_tensor(
        teacher_scores, dtype=torch.float32)
    fs = torch.as_tensor(list(first_stage_ranks), dtype=torch.float32, device=ss.device)
    ts = ts.to(ss.device).float()

    higher = fs.unsqueeze(1) < fs.unsqueeze(0)               # [i,j] True if rank_i < rank_j
    tmargin = ts.unsqueeze(0) - ts.unsqueeze(1)              # [i,j] = teacher_j - teacher_i
    justified = tmargin >= justify_margin
    mask = (higher & (~justified)).float()
    sdiff = ss.unsqueeze(0) - ss.unsqueeze(1)                # [i,j] = student_j - student_i
    penalty = torch.relu(sdiff) * mask
    denom = mask.sum().clamp(min=1.0)
    return penalty.sum() / denom
