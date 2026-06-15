"""WebFAQ / WebFAQ 2.0 hard-negative loader (pure stdlib core, no network by default).

WebFAQ 2.0 ships a hard-negative dataset (~1.25M queries x 20 langs, up to 200 negatives/query
with cross-encoder scores) usable for MNRL + MarginMSE. This module makes those hard negatives a
first-class v5 training source by converting one WebFAQ2 record into BOTH:

- **embedder triplets** ``(query, positive, negative, teacher_margin)`` for MarginMSE/MNRL, and
- **reranker candidate lists** with **listwise** teacher scores (+ softmax target).

It **fails closed**: a missing local file (without ``--download-hf``), an unknown/absent license,
or a record without a positive cross-encoder score is rejected rather than silently guessed. Local
JSONL is the default path; the Hugging Face loader is optional and lazily imported only behind
``--download-hf`` (never in tests).
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .v5_data_mixer import is_unknown_license

DEFAULT_MIN_MARGIN = 2.0
DEFAULT_MAX_NEGATIVES = 32
DEFAULT_FALSE_NEGATIVE_MARGIN = 0.5     # negative within this of the positive => likely a false neg
SOURCE = "webfaq2"
HARDNEG_SOURCE = "webfaq2_hardneg"
MARGIN_BUCKETS = ("<0", "0-1", "1-2", "2-3", "3-4", "4-5", ">=5")


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _doc_id(text: str) -> str:
    return "d" + hashlib.blake2b(text.encode("utf-8"), digest_size=10).hexdigest()


def _softmax(scores: List[float]) -> List[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    tot = sum(exps) or 1.0
    return [e / tot for e in exps]


def _margin_bucket(margin: float) -> str:
    if margin < 0:
        return "<0"
    if margin >= 5:
        return ">=5"
    return f"{int(margin)}-{int(margin) + 1}"


def normalize_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map a WebFAQ2 record (several possible shapes) to a canonical record."""
    q = raw.get("query") or raw.get("question")
    pos = (raw.get("positive") or raw.get("positive_document") or raw.get("answer")
           or raw.get("document"))
    pos_score = (raw.get("positive_score") or raw.get("positive_cross_encoder_score")
                 or raw.get("positive_ce_score"))
    negs_raw = raw.get("negatives") or raw.get("hard_negatives") or []
    par_scores = raw.get("negative_scores") or raw.get("negative_cross_encoder_scores")
    negatives: List[Dict[str, Any]] = []
    for i, n in enumerate(negs_raw):
        if isinstance(n, dict):
            negatives.append({
                "document": n.get("document") or n.get("text") or n.get("passage"),
                "cross_encoder_score": n.get("cross_encoder_score",
                                             n.get("score", n.get("ce_score"))),
                "title": n.get("title"), "url": n.get("url") or n.get("source_url")})
        else:
            negatives.append({
                "document": n,
                "cross_encoder_score": (par_scores[i] if isinstance(par_scores, list)
                                        and i < len(par_scores) else None),
                "title": None, "url": None})
    return {
        "query_id": raw.get("query_id") or (_doc_id(q) if isinstance(q, str) else None),
        "query": q, "positive": pos, "positive_score": pos_score, "negatives": negatives,
        "language": raw.get("language") or raw.get("lang"),
        "license": raw.get("license"),
        "title": raw.get("title"), "source_url": raw.get("source_url") or raw.get("url"),
    }


def validate_record(rec: Dict[str, Any], idx: int) -> List[str]:
    errs: List[str] = []
    if not isinstance(rec.get("query"), str) or not rec["query"].strip():
        errs.append(f"record[{idx}]: 'query' must be a non-empty string")
    if not isinstance(rec.get("positive"), str) or not rec["positive"].strip():
        errs.append(f"record[{idx}]: 'positive' must be a non-empty string")
    if not _is_number(rec.get("positive_score")):
        errs.append(f"record[{idx}]: 'positive_score' (positive cross-encoder score) is required")
    if is_unknown_license(rec.get("license")):
        errs.append(f"record[{idx}]: unknown/absent license (fail closed)")
    if not isinstance(rec.get("negatives"), list) or not rec["negatives"]:
        errs.append(f"record[{idx}]: 'negatives' must be a non-empty list")
    return errs


def filter_negatives(rec: Dict[str, Any], *, min_margin: float, max_negatives: int,
                     false_negative_margin: float) -> Dict[str, Any]:
    """Keep clearly-negative hard negatives; drop too-close ones (and flag false negatives).

    A negative is kept iff (positive_score - negative_score) >= ``min_margin``. Dropped negatives
    are classified ``false_negative`` when the margin is within ``false_negative_margin`` of the
    positive (likely actually relevant) else ``insufficient_margin``. Kept negatives are sorted
    HARDEST-first (smallest margin >= min_margin) and capped to ``max_negatives`` deterministically."""
    pos_score = float(rec["positive_score"])
    kept: List[Dict[str, Any]] = []
    dropped_false: List[Dict[str, Any]] = []
    dropped_easy: List[Dict[str, Any]] = []
    all_margins: List[float] = []
    for n in rec["negatives"]:
        doc = n.get("document")
        sc = n.get("cross_encoder_score")
        if not isinstance(doc, str) or not doc.strip() or not _is_number(sc):
            continue
        margin = pos_score - float(sc)
        all_margins.append(margin)
        cand = {"document": doc, "cross_encoder_score": float(sc), "margin": margin,
                "title": n.get("title"), "url": n.get("url")}
        if margin >= min_margin:
            kept.append(cand)
        elif margin <= false_negative_margin:
            dropped_false.append(cand)
        else:
            dropped_easy.append(cand)
    # hardest-first, deterministic tie-break by doc id
    kept.sort(key=lambda c: (c["margin"], _doc_id(c["document"])))
    capped = kept[:max_negatives]
    return {"kept": capped, "kept_total": len(kept), "capped_out": max(0, len(kept) - len(capped)),
            "dropped_false_negatives": dropped_false, "dropped_insufficient_margin": dropped_easy,
            "all_margins": all_margins}


def to_embedder_triplets(rec: Dict[str, Any], kept: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pos_score = float(rec["positive_score"])
    out = []
    for n in kept:
        row = {"query": rec["query"], "positive": rec["positive"], "negative": n["document"],
               "teacher_margin": round(pos_score - n["cross_encoder_score"], 6),
               "positive_score": pos_score, "negative_score": n["cross_encoder_score"],
               "language": rec.get("language"), "license": rec["license"], "source": SOURCE}
        if rec.get("source_url"):
            row["source_url"] = rec["source_url"]
        if rec.get("title"):
            row["title"] = rec["title"]
        out.append(row)
    return out


def to_reranker_list(rec: Dict[str, Any], kept: List[Dict[str, Any]]) -> Dict[str, Any]:
    pos_doc = rec["positive"]
    cands = [{"doc_id": _doc_id(pos_doc), "text": pos_doc,
              "teacher_score": float(rec["positive_score"]), "is_positive": True, "source": SOURCE}]
    for n in kept:
        cands.append({"doc_id": _doc_id(n["document"]), "text": n["document"],
                      "teacher_score": n["cross_encoder_score"], "is_positive": False,
                      "source": HARDNEG_SOURCE})
    soft = _softmax([c["teacher_score"] for c in cands])
    for c, s in zip(cands, soft):
        c["teacher_softmax_target"] = round(s, 6)
    return {"query_id": rec["query_id"], "query": rec["query"],
            "language": rec.get("language"), "license": rec["license"],
            "positive_doc_id": _doc_id(pos_doc), "candidates": cands, "source": SOURCE}


def import_webfaq2(raw_records: Iterable[Dict[str, Any]], *, language: str,
                   min_margin: float = DEFAULT_MIN_MARGIN,
                   max_negatives: int = DEFAULT_MAX_NEGATIVES,
                   false_negative_margin: float = DEFAULT_FALSE_NEGATIVE_MARGIN) -> Dict[str, Any]:
    """Validate, language-filter, margin-filter, and convert to triplets + reranker lists."""
    errors: List[str] = []
    triplets: List[Dict[str, Any]] = []
    lists: List[Dict[str, Any]] = []
    negs_per_query: List[int] = []
    margin_hist = {b: 0 for b in MARGIN_BUCKETS}
    dropped_false = dropped_easy = capped_out = 0
    skipped_language = 0
    by_license: Dict[str, int] = {}
    by_language: Dict[str, int] = {}

    norm = [normalize_record(r) for r in raw_records if isinstance(r, dict)]
    for i, rec in enumerate(norm):
        if language and rec.get("language") not in (language, None):
            skipped_language += 1
            continue
        rec_errs = validate_record(rec, i)
        if rec_errs:
            errors += rec_errs
            continue
        f = filter_negatives(rec, min_margin=min_margin, max_negatives=max_negatives,
                             false_negative_margin=false_negative_margin)
        for m in f["all_margins"]:
            margin_hist[_margin_bucket(m)] += 1
        dropped_false += len(f["dropped_false_negatives"])
        dropped_easy += len(f["dropped_insufficient_margin"])
        capped_out += f["capped_out"]
        if not f["kept"]:
            continue                       # no usable hard negative after filtering
        triplets += to_embedder_triplets(rec, f["kept"])
        lists.append(to_reranker_list(rec, f["kept"]))
        negs_per_query.append(len(f["kept"]))
        lic = str(rec["license"]).strip().lower()
        by_license[lic] = by_license.get(lic, 0) + 1
        lang = str(rec.get("language"))
        by_language[lang] = by_language.get(lang, 0) + 1

    status = "fail" if errors or not lists else "pass"
    report = {
        "status": status,
        "errors": errors[:50],
        "error_count": len(errors),
        "params": {"language": language, "min_cross_encoder_margin": min_margin,
                   "max_negatives_per_query": max_negatives,
                   "false_negative_margin": false_negative_margin},
        "imported_queries": len(lists),
        "skipped_other_language": skipped_language,
        "embedder_triplets": len(triplets),
        "reranker_lists": len(lists),
        "negatives_per_query": {
            "total": sum(negs_per_query),
            "avg": round(sum(negs_per_query) / len(negs_per_query), 3) if negs_per_query else 0,
            "min": min(negs_per_query) if negs_per_query else 0,
            "max": max(negs_per_query) if negs_per_query else 0,
        },
        "margin_distribution": margin_hist,
        "dropped_false_negatives": dropped_false,
        "dropped_insufficient_margin": dropped_easy,
        "capped_out_negatives": capped_out,
        "by_license": dict(sorted(by_license.items())),
        "by_language": dict(sorted(by_language.items())),
    }
    return {"report": report, "triplets": triplets, "reranker_lists": lists}


def load_local_jsonl(path: str) -> List[Dict[str, Any]]:
    import pathlib
    # split("\n"): WebFAQ text carries U+2028/U+2029 which splitlines() over-splits.
    return [__import__("json").loads(ln)
            for ln in pathlib.Path(path).read_text(encoding="utf-8").split("\n") if ln.strip()]


def load_from_hf(dataset: str, language: str, split: str = "train",
                 limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """OPTIONAL Hugging Face loader — lazily imports `datasets`. Only reached behind --download-hf."""
    from datasets import load_dataset  # noqa: E402  (lazy; never imported in tests)
    ds = load_dataset(dataset, language, split=split) if language else load_dataset(dataset, split=split)
    rows = []
    for i, r in enumerate(ds):
        if limit is not None and i >= limit:
            break
        rows.append(dict(r))
    return rows
