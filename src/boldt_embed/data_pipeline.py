"""German multi-domain training-data pipeline (pure stdlib).

Replaces the Wikipedia-only path that overfit in v1. Produces a leakage-aware,
domain-balanced, teacher-score-ready *candidate* JSONL. Everything here is deterministic
and standard-library only — no ML deps, no network — so it runs in the unit-test gate and
behind ``--dry-run``.

Candidate schema (one JSON object per (query, document) pair):

    {
      "query_id": "...", "doc_id": "...",
      "query": "...", "document": "...",
      "positive": true,
      "source": "mMARCO-de|clips-mqa-de|synthetic|...",
      "domain": "web|faq|admin|legal_adversarial|wiki|...",
      "license": "...",
      "metadata": {...}
    }
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

CANDIDATE_REQUIRED = ("query_id", "doc_id", "query", "document", "positive",
                      "source", "domain", "license")


# ------------------------------------------------------------------- normalization/IO
def normalize_text(s: str) -> str:
    """NFC-normalize and collapse all whitespace to single spaces. Deterministic."""
    return " ".join(unicodedata.normalize("NFC", s).split())


def stable_text_hash(text: str, normalize: bool = True, length: int = 16) -> str:
    """Deterministic content hash. With ``normalize`` (default), text differing only in
    Unicode form or whitespace hashes identically — which is what makes leakage filtering
    and dedup robust across sources."""
    t = normalize_text(text) if normalize else text
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[:length]


def stable_pair_id(query: str, document: str) -> str:
    """Deterministic id for a (query, document) pair."""
    h = stable_text_hash(query) + "::" + stable_text_hash(document)
    return "p" + hashlib.sha256(h.encode("utf-8")).hexdigest()[:16]


def detect_language_hint_simple(text: str) -> str:
    """Cheap deterministic heuristic: 'de' if German signals (umlauts/ß or common German
    function words) are present, else 'unknown'. Not a real language ID — a guard rail."""
    low = text.lower()
    if any(ch in low for ch in "äöüß"):
        return "de"
    words = set(low.replace(",", " ").replace(".", " ").split())
    german_markers = {"der", "die", "das", "und", "ist", "nicht", "für", "mit", "ein",
                      "eine", "den", "dem", "von", "zu", "auf", "wird", "werden"}
    return "de" if len(words & german_markers) >= 2 else "unknown"


def validate_candidate_record(row: Any) -> List[str]:
    """Return a list of problems with a *training candidate* row (never raises)."""
    errors: List[str] = []
    if not isinstance(row, dict):
        return ["candidate must be a JSON object"]
    for key in ("query_id", "doc_id", "query", "document", "source", "domain", "license"):
        v = row.get(key)
        if not isinstance(v, str) or not v.strip():
            errors.append(f"missing/empty required string field '{key}'")
    if not isinstance(row.get("positive"), bool):
        errors.append("'positive' must be a bool")
    if "metadata" in row and not isinstance(row["metadata"], dict):
        errors.append("'metadata' must be an object")
    return errors


def stream_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def normalize_record(row: Dict[str, Any], *, default_source: Optional[str] = None,
                     default_domain: Optional[str] = None,
                     default_license: Optional[str] = None) -> Dict[str, Any]:
    """Coerce a raw row into the canonical candidate schema: NFC/whitespace-normalize the
    texts, derive stable ids from content when missing, and apply field defaults."""
    query = normalize_text(str(row.get("query", "")))
    document = normalize_text(str(row.get("document", "")))
    out = {
        "query_id": str(row.get("query_id") or ("q" + stable_text_hash(query))),
        "doc_id": str(row.get("doc_id") or ("d" + stable_text_hash(document))),
        "query": query,
        "document": document,
        "positive": bool(row.get("positive", True)),
        "source": str(row.get("source") or default_source or "unknown"),
        "domain": str(row.get("domain") or default_domain or "unknown"),
        "license": str(row.get("license") or default_license or "unknown"),
        "metadata": dict(row.get("metadata") or {}),
    }
    return out


# ------------------------------------------------------------------- selection/filter
def deduplicate_by_text_hash(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop duplicate (query, document) pairs, keeping first occurrence. Deterministic."""
    seen = set()
    out = []
    for r in rows:
        key = stable_pair_id(r.get("query", ""), r.get("document", ""))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def domain_balanced_sample(rows: Sequence[Dict[str, Any]], max_per_domain: int
                           ) -> List[Dict[str, Any]]:
    """Cap each domain at ``max_per_domain``, preserving input order. Deterministic."""
    counts: Dict[str, int] = {}
    out = []
    for r in rows:
        dom = str(r.get("domain", "unknown"))
        if counts.get(dom, 0) < max_per_domain:
            out.append(r)
            counts[dom] = counts.get(dom, 0) + 1
    return out


def sample_to_domain_targets(rows: Sequence[Dict[str, Any]], targets: Dict[str, int],
                             seed: int = 0) -> List[Dict[str, Any]]:
    """Deterministically sample up to ``targets[domain]`` rows per domain (seeded shuffle).
    Domains absent from ``targets`` keep all their rows. Returns rows grouped by domain in
    sorted-domain order for reproducibility."""
    import random as _random
    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_domain.setdefault(str(r.get("domain", "unknown")), []).append(r)
    out: List[Dict[str, Any]] = []
    for dom in sorted(by_domain):
        group = list(by_domain[dom])
        _random.Random(seed).shuffle(group)
        cap = targets.get(dom)
        out.extend(group if cap is None else group[:cap])
    return out


def filter_leakage_against_eval_texts(rows: Sequence[Dict[str, Any]],
                                      eval_texts: Iterable[str],
                                      fields: Tuple[str, ...] = ("query", "document"),
                                      ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Drop any candidate whose query OR document text matches an eval-corpus text
    (normalized hash). Returns (kept_rows, stats). This is the hard wall that keeps public
    benchmark test data eval-only: candidates leaking eval text are removed."""
    banned = {stable_text_hash(t) for t in eval_texts}
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for r in rows:
        if any(stable_text_hash(str(r.get(f, ""))) in banned for f in fields):
            dropped += 1
        else:
            kept.append(r)
    return kept, {"input": len(rows), "kept": len(kept), "dropped": dropped,
                  "eval_texts": len(banned)}


def domain_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in rows:
        counts[str(r.get("domain", "unknown"))] = counts.get(str(r.get("domain", "unknown")), 0) + 1
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------- v3 candidate rows
# v3 rows carry FULL provenance end-to-end (the v2 bug dropped license at the cache stage).
def build_v3_candidate_row(query: str, document: str, *, source_id: str, domain: str,
                           license: str, license_origin: str, allowed_for_training: bool,
                           synthetic: bool, source_url: Optional[str] = None) -> Dict[str, Any]:
    """A v3 (query, document) candidate with explicit provenance. ``synthetic`` is carried so
    domain-quality can separate real vs synthetic; ``record_type='pair'``."""
    q = normalize_text(str(query))
    d = normalize_text(str(document))
    row = {
        "record_type": "pair",
        "query_id": "q" + stable_text_hash(q), "doc_id": "d" + stable_text_hash(d),
        "query": q, "document": d, "positive": True,
        "source_id": source_id, "source": source_id, "domain": domain,
        "license": license, "license_origin": license_origin,
        "allowed_for_training": bool(allowed_for_training), "synthetic": bool(synthetic),
        "text_hash": stable_text_hash(d), "pair_hash": stable_pair_id(q, d),
    }
    if source_url:
        row["source_url"] = source_url
    return row


def build_v3_passage_record(document: str, *, source_id: str, domain: str, license: str,
                            license_origin: str, allowed_for_training: bool, synthetic: bool,
                            source_url: Optional[str] = None) -> Dict[str, Any]:
    """A DOCUMENT-ONLY passage (no query). For document-only corpora we never fabricate a
    query/doc pair — a later generator emits synthetic pairs and marks them. ``record_type=
    'passage'``; no query_id/pair_hash."""
    d = normalize_text(str(document))
    row = {
        "record_type": "passage",
        "doc_id": "d" + stable_text_hash(d), "document": d,
        "source_id": source_id, "source": source_id, "domain": domain,
        "license": license, "license_origin": license_origin,
        "allowed_for_training": bool(allowed_for_training), "synthetic": bool(synthetic),
        "text_hash": stable_text_hash(d),
    }
    if source_url:
        row["source_url"] = source_url
    return row


def quota_report(rows: Sequence[Dict[str, Any]], targets: Dict[str, int],
                 real_domains: Sequence[str] = ()) -> Dict[str, Any]:
    """Per-domain achieved-vs-target with real/synthetic split and a missed-quota list.
    For real domains, only NON-synthetic pairs count toward the target (the v2 lesson)."""
    real_domains = set(real_domains)
    by_dom: Dict[str, Dict[str, int]] = {}
    for r in rows:
        dom = str(r.get("domain", "unknown"))
        b = by_dom.setdefault(dom, {"total": 0, "real": 0, "synthetic": 0})
        b["total"] += 1
        b["synthetic" if r.get("synthetic") else "real"] += 1
    out: Dict[str, Any] = {"by_domain": {}, "missed": []}
    for dom, target in sorted(targets.items()):
        b = by_dom.get(dom, {"total": 0, "real": 0, "synthetic": 0})
        counted = b["real"] if dom in real_domains else b["total"]
        achieved = counted >= target
        out["by_domain"][dom] = {"target": target, "total": b["total"], "real": b["real"],
                                 "synthetic": b["synthetic"], "counted_toward_target": counted,
                                 "achieved": achieved}
        if not achieved:
            out["missed"].append({"domain": dom, "target": target, "counted": counted,
                                  "real_domain": dom in real_domains})
    return out
