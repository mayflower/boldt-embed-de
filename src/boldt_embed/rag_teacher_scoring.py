"""Teacher-score FULL RAG candidate lists for listwise distillation + high-precision labels.

Every (query, document) candidate is scored by the Qwen3 reranker teacher (and optionally the
embedding teacher's cosine), then each list is annotated with teacher_rank, a listwise
softmax target, and a high-precision label policy:

- **gold positives** (doc_id in positive_doc_ids) stay positive (label 1);
- a non-gold candidate the teacher scores HIGH (>= positive_threshold) is a **teacher-only
  positive** → marked ``uncertain`` and used for **listwise distillation only** (label null
  unless ``use_teacher_only_positives``);
- a candidate scored too CLOSE to the positive band is ``uncertain`` (label null) — never a
  hard BCE negative;
- a **hard negative** must be clearly below the positive band (label 0).

The annotation/labeling/summary layer is pure stdlib (testable on fixtures with precomputed
scores); only the actual teacher inference lazy-imports torch.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional, Sequence

from .reranker_modern import v3_label

POSITIVE_THRESHOLD = 4.0
HARD_NEG_MARGIN = 2.0      # a clear negative scores <= positive_threshold - margin


def softmax(scores: Sequence[float], temperature: float = 1.0) -> List[float]:
    if not scores:
        return []
    t = max(temperature, 1e-6)
    m = max(scores)
    exps = [math.exp((s - m) / t) for s in scores]
    z = sum(exps) or 1.0
    return [e / z for e in exps]


def _teacher_score(c: Dict[str, Any]) -> Optional[float]:
    s = c.get("teacher_score")
    if s is None:
        s = c.get("reranker_score")
    return None if s is None else float(s)


def annotate_list(row: Dict[str, Any], *, positive_threshold: float = POSITIVE_THRESHOLD,
                  hard_neg_margin: float = HARD_NEG_MARGIN, temperature: float = 1.0,
                  use_teacher_only_positives: bool = False) -> Dict[str, Any]:
    """Annotate one candidate-list row in place-ish (returns a new row): adds teacher_rank,
    teacher_softmax_target, high_precision_positive, uncertain, and the policy label per
    candidate. Candidates must already carry ``teacher_score`` (set by scoring or fixtures)."""
    cands = [dict(c) for c in row.get("candidates", [])]
    positives = set(row.get("positive_doc_ids") or [])
    scores = [(_teacher_score(c) if _teacher_score(c) is not None else float("-inf")) for c in cands]

    # listwise softmax target over teacher scores (missing -> very low)
    target = softmax(scores, temperature)
    # teacher_rank: 1-based by teacher score desc, deterministic (doc_id tie-break)
    order = sorted(range(len(cands)), key=lambda i: (-scores[i], cands[i].get("doc_id", "")))
    rank_of = {i: r + 1 for r, i in enumerate(order)}

    for i, c in enumerate(cands):
        ts = _teacher_score(c)
        gold = c.get("doc_id") in positives
        c["teacher_rank"] = rank_of[i]
        c["teacher_softmax_target"] = round(target[i], 6)
        if gold:
            c["label"] = 1
            c["high_precision_positive"] = ts is not None and ts >= positive_threshold
            c["uncertain"] = False
        else:
            c["high_precision_positive"] = False
            lab = v3_label(ts, positive_threshold, hard_neg_margin)
            if lab == 1:                       # teacher-only positive (teacher disagrees w/ gold)
                c["uncertain"] = True
                c["label"] = 1 if use_teacher_only_positives else None
            elif lab == 0:                     # clear hard negative
                c["uncertain"] = False
                c["label"] = 0
            else:                              # too close to positive -> uncertain
                c["uncertain"] = True
                c["label"] = None
    out = dict(row)
    out["candidates"] = cands
    return out


def annotate_lists(rows: Sequence[Dict[str, Any]], **kw) -> List[Dict[str, Any]]:
    return [annotate_list(r, **kw) for r in rows]


def _median(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


def summarize(rows: Sequence[Dict[str, Any]], positive_threshold: float = POSITIVE_THRESHOLD
              ) -> Dict[str, Any]:
    """Teacher-score separation by domain/source, pos/neg medians, uncertain fraction,
    candidate-source quality, and teacher-vs-gold disagreement examples."""
    pos_by_dom: Dict[str, List[float]] = {}
    neg_by_dom: Dict[str, List[float]] = {}
    by_source: Dict[str, List[float]] = {}
    n_cand = n_uncertain = n_pos = n_neg = 0
    disagreements: List[Dict[str, Any]] = []
    for r in rows:
        for c in r.get("candidates", []):
            n_cand += 1
            ts = _teacher_score(c)
            dom = str(c.get("domain") or r.get("domain") or "unknown")
            src = str(c.get("candidate_source") or "unknown")
            if ts is not None:
                by_source.setdefault(src, []).append(ts)
            if c.get("uncertain"):
                n_uncertain += 1
            if c.get("label") == 1:
                n_pos += 1
                if ts is not None:
                    pos_by_dom.setdefault(dom, []).append(ts)
            elif c.get("label") == 0:
                n_neg += 1
                if ts is not None:
                    neg_by_dom.setdefault(dom, []).append(ts)
            gold = c.get("doc_id") in set(r.get("positive_doc_ids") or [])
            if gold and ts is not None and ts < positive_threshold:
                disagreements.append({"query_id": r.get("query_id"), "doc_id": c.get("doc_id"),
                                      "kind": "gold_low_teacher", "teacher_score": ts})
            elif (not gold) and ts is not None and ts >= positive_threshold:
                disagreements.append({"query_id": r.get("query_id"), "doc_id": c.get("doc_id"),
                                      "kind": "teacher_only_positive", "teacher_score": ts})
    sep = {}
    for dom in sorted(set(pos_by_dom) | set(neg_by_dom)):
        pm, nm = _median(pos_by_dom.get(dom, [])), _median(neg_by_dom.get(dom, []))
        sep[dom] = {"pos_median": pm, "neg_median": nm,
                    "separation": round(pm - nm, 4) if (pm is not None and nm is not None) else None}
    return {
        "n_lists": len(rows), "n_candidates": n_cand,
        "positives": n_pos, "negatives": n_neg, "uncertain": n_uncertain,
        "uncertain_fraction": round(n_uncertain / n_cand, 4) if n_cand else 0.0,
        "separation_by_domain": sep,
        "candidate_source_quality": {s: {"n": len(v), "median_teacher": _median(v),
                                         "pct_ge_threshold": round(
                                             100 * sum(1 for x in v if x >= positive_threshold) / len(v), 1)}
                                     for s, v in sorted(by_source.items())},
        "teacher_disagreements": disagreements,
        "n_teacher_disagreements": len(disagreements),
    }


# ----------------------------------------------------------------- ML layer (lazy import)
def score_lists_with_teacher(rows: Sequence[Dict[str, Any]], teacher_cfg, mode: str = "reranker",
                             device: Optional[str] = None) -> List[Dict[str, Any]]:
    """Score every (query, doc) candidate with the Qwen3 reranker teacher (and optional embedding
    cosine). Returns rows with teacher_score (+ embedding_score) on each candidate. ML-only."""
    from . import teacher as T
    rr = T.load_reranker_teacher(teacher_cfg.reranker_teacher, device=device)
    emb = T.load_embedding_teacher(teacher_cfg.embedding_teacher, device=device) \
        if mode == "both" else None
    out = []
    for r in rows:
        q = r.get("query", "")
        pairs = [(q, c.get("text", "")) for c in r.get("candidates", [])]
        rr_scores = T.score_pairs_with_reranker_teacher(rr, pairs, teacher_cfg.reranker_teacher)
        emb_scores = (T.score_pairs_with_embedding_teacher(emb, pairs, teacher_cfg.embedding_teacher)
                      if emb is not None else [None] * len(pairs))
        cands = []
        for c, rs, es in zip(r.get("candidates", []), rr_scores, emb_scores):
            c = dict(c)
            c["teacher_score"] = float(rs)
            if es is not None:
                c["embedding_score"] = float(es)
            cands.append(c)
        nr = dict(r)
        nr["candidates"] = cands
        out.append(nr)
    return out
