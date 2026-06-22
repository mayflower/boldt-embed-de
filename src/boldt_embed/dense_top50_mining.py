"""Mine dense-specific hard negatives to improve WebFAQ Recall@50 (pure stdlib, no ML).

Dense-v6 has Recall@100 0.964 but Recall@50 0.883: the positive is often in the top-100/200 but
BELOW rank 50. These negatives teach the dense model to demote the docs that currently outrank the
positive, pulling it into the top-50. **DENSE-ONLY — no reranker training.**

Mining categories (per query whose positive is at dense rank 51..200):
  - `dense_top50_false_positive` — docs the dense model ranks ABOVE the positive (the blockers).
  - `teacher`                    — high-dense-rank docs the teacher confirms are non-relevant.
  - `bm25`                       — high-BM25 lexical confusions that are not relevant.
False-negative veto: drop any candidate whose teacher score is within ``veto_margin`` of the
positive's (it may actually be relevant — never train it as a negative).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

SRC_DENSE_FP = "dense_top50_false_positive"
SRC_TEACHER = "teacher"
SRC_BM25 = "bm25"


def positive_rank(dense_ranked: Sequence[str], positive_doc_id: str) -> Optional[int]:
    """1-indexed rank of the positive in the dense ranking, or None if absent."""
    for i, d in enumerate(dense_ranked):
        if d == positive_doc_id:
            return i + 1
    return None


def mine_query(qrec: Dict[str, Any], corpus: Dict[str, str], *, positives: Optional[Set[str]] = None,
               teacher_scores: Optional[Dict[str, float]] = None, top50: int = 50, window: int = 200,
               veto_margin: float = 2.0, max_negatives: int = 20) -> Optional[Dict[str, Any]]:
    """Mine hard negatives for ONE query. Returns the record (schema below) or None if the query is
    not a rank-51..window target case. Deterministic. ``teacher_scores`` maps doc_id -> teacher score
    for this query (overrides ``qrec['teacher_scores']``)."""
    dense = list(qrec.get("dense_ranked") or [])
    pid = qrec.get("positive_doc_id")
    pos: Set[str] = set(positives or ([pid] if pid else []))
    rank_of = {d: i + 1 for i, d in enumerate(dense)}
    pr = rank_of.get(pid)
    # TARGET: positive present but below top-50, within the window (the Recall@50 cases).
    if pr is None or pr <= top50 or pr > window:
        return None
    ts = teacher_scores if teacher_scores is not None else (qrec.get("teacher_scores") or {})
    pos_ts = ts.get(pid)

    cands: List = []                       # (dense_rank_or_None, doc_id, source) — source priority
    seen: Set[str] = set()
    # a) dense false positives: docs ranked ABOVE the positive (these block top-50 recall)
    for d in dense[:pr - 1]:
        if d in pos or d in seen:
            continue
        seen.add(d); cands.append((rank_of[d], d, SRC_DENSE_FP))
    # b) teacher-confirmed hard negatives in the window BELOW the positive
    for d in dense[pr:window]:
        if d in pos or d in seen:
            continue
        s = ts.get(d)
        if s is not None and pos_ts is not None and (pos_ts - s) >= veto_margin:
            seen.add(d); cands.append((rank_of[d], d, SRC_TEACHER))
    # c) BM25 lexical confusions (if provided)
    for d in (qrec.get("bm25_ranked") or [])[:window]:
        if d in pos or d in seen:
            continue
        seen.add(d); cands.append((rank_of.get(d), d, SRC_BM25))

    negs: List[Dict[str, Any]] = []
    vetoed = 0
    for drank, d, src in cands:
        nts = ts.get(d)
        margin = (pos_ts - nts) if (pos_ts is not None and nts is not None) else None
        if margin is not None and margin < veto_margin:
            vetoed += 1                    # false-negative veto: too close to the positive
            continue
        negs.append({"doc_id": d, "text": corpus.get(d, ""), "negative_rank_v6": drank,
                     "source": src, "teacher_score": nts,
                     "margin_to_positive": (round(margin, 4) if margin is not None else None)})
    # keep the highest-dense-rank blockers first (smallest rank); unranked (bm25-only) last
    negs.sort(key=lambda n: (n["negative_rank_v6"] is None,
                             n["negative_rank_v6"] if n["negative_rank_v6"] is not None else 1 << 30,
                             n["doc_id"]))
    negs = negs[:max_negatives]
    return {
        "query_id": qrec.get("query_id"), "query": qrec.get("query", ""),
        "positive_doc_id": pid, "positive": corpus.get(pid, ""), "positive_rank_v6": pr,
        "negatives": negs, "domain": qrec.get("domain", "unknown"),
        "source": qrec.get("source", "dense_top50_mining"), "_vetoed": vetoed,
    }


def _margin_bucket(m: float) -> str:
    if m < 0:
        return "<0"
    if m >= 6:
        return ">=6"
    return f"{int(m)}-{int(m) + 1}"


def mine_set(qrecs: Sequence[Dict[str, Any]], corpus: Dict[str, str], *,
             qrels: Optional[Dict[str, Set[str]]] = None, teacher_scores: Optional[Dict] = None,
             top50: int = 50, window: int = 200, veto_margin: float = 2.0, max_negatives: int = 20,
             leakage_check: bool = True) -> Dict[str, Any]:
    """Mine the whole set. ``teacher_scores`` may be a nested {query_id: {doc_id: score}} map.
    Public-eval-leaking queries are excluded (no public-eval train leakage). Returns
    {records, report}. Pure stdlib."""
    from .v5_data_mixer import leakage_reason
    records: List[Dict[str, Any]] = []
    leaked: List[str] = []
    r51_100 = r101_200 = veto_total = 0
    margins: List[float] = []
    by_source: Dict[str, int] = {}
    n_seen = 0
    for q in qrecs:
        if leakage_check and leakage_reason(q):
            leaked.append(str(q.get("query_id")))
            continue
        n_seen += 1
        qid = str(q.get("query_id"))
        pos = (qrels or {}).get(qid)
        ts = None
        if teacher_scores is not None:
            ts = teacher_scores.get(qid) if qid in teacher_scores else teacher_scores
        rec = mine_query(q, corpus, positives=pos, teacher_scores=ts, top50=top50, window=window,
                         veto_margin=veto_margin, max_negatives=max_negatives)
        if rec is None:
            continue
        veto_total += rec.pop("_vetoed", 0)
        pr = rec["positive_rank_v6"]
        if pr <= 100:
            r51_100 += 1
        else:
            r101_200 += 1
        for ng in rec["negatives"]:
            by_source[ng["source"]] = by_source.get(ng["source"], 0) + 1
            if ng["margin_to_positive"] is not None:
                margins.append(ng["margin_to_positive"])
        records.append(rec)

    hist: Dict[str, int] = {}
    for m in margins:
        hist[_margin_bucket(m)] = hist.get(_margin_bucket(m), 0) + 1
    n_neg = sum(len(r["negatives"]) for r in records)
    report = {
        "queries_seen": n_seen, "queries_mined": len(records),
        "positive_rank_51_100": r51_100, "positive_rank_101_200": r101_200,
        "total_negatives": n_neg,
        "avg_negatives_per_query": round(n_neg / len(records), 3) if records else 0.0,
        "false_negative_veto_count": veto_total,
        "negatives_by_source": dict(sorted(by_source.items())),
        "teacher_margin_distribution": dict(sorted(hist.items())),
        "leakage_excluded": len(leaked),
        "params": {"top50": top50, "window": window, "veto_margin": veto_margin,
                   "max_negatives": max_negatives},
    }
    return {"records": records, "report": report}
