"""v5 RAG data mixer (pure stdlib, no ML, no network).

Builds a **domain-balanced**, leakage-safe German RAG training mix from heterogeneous input
sources so v5 cannot silently become another FAQ-only run (the v4 failure mode). The mixer:

- validates every row against the v5 input schema;
- HARD-FAILS on unknown license, on public-benchmark/eval leakage, and on a mixture that is too
  FAQ-heavy (FAQ share over a cap) or too FAQ-poor (non-FAQ share under a floor);
- samples **deterministically** (blake2b stable key + round-robin over sorted domains), so the
  same inputs always yield the same mixture;
- emits a report proving non-FAQ coverage *before* any teacher scoring.

Row schema (one JSON object per input line)::

    {
      "source_id": "...",                # required
      "domain": "faq_real|qa_passage_non_eval|web_nonfaq|long_doc_chunks|local_rag|german_stress",
      "query": "...",                    # required
      "document": "...",                 # required
      "title": "...",                    # optional
      "answer": "...",                   # optional
      "license": "CC-BY-4.0",            # required; must be a known license
      "source_url": "...",               # optional
      "synthetic_query": true,           # required bool
      "generation_method": "...",        # optional
      "eval_only": false,                # optional bool (default false); true => leakage
      "public_benchmark": false          # optional bool (default false); true => leakage
    }

Text-level provenance (e.g. "is this passage actually GermanQuAD test text?") is NOT decided
here — that remains the job of the dedicated leakage index (`scripts/run_full_leakage_scan.py`).
This mixer enforces the *declared* ``eval_only`` / ``public_benchmark`` flags plus public-
benchmark tokens in ``source_id`` / ``source_url`` / ``domain``.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Tuple

from .data import ALLOWED_LICENSES as _BASE_LICENSES
from .v5_rag_config import is_public_benchmark

FAQ_DOMAIN = "faq_real"
# The v5 training domains (mirror configs/experiments/v5_small_rag.json train_domains).
TRAIN_DOMAINS: Tuple[str, ...] = (
    "faq_real", "qa_passage_non_eval", "web_nonfaq",
    "long_doc_chunks", "local_rag", "german_stress",
)
NONFAQ_DOMAINS = tuple(d for d in TRAIN_DOMAINS if d != FAQ_DOMAIN)

# Known training licenses. Extends data.ALLOWED_LICENSES with the inherited-synthetic tag the
# acquisition step uses for teacher-generated queries over a permissively-licensed source doc.
V5_ALLOWED_LICENSES = frozenset(_BASE_LICENSES | {"synthetic-inherits-source"})

REQUIRED_STR_FIELDS = ("source_id", "domain", "query", "document", "license")


def normalize_license(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def is_unknown_license(value: Any) -> bool:
    """True if the license is missing, empty, "unknown", or not in the v5 allowlist."""
    s = normalize_license(value)
    return s == "" or s == "unknown" or s not in V5_ALLOWED_LICENSES


def classify_query_style(query: str) -> str:
    """Coarse, deterministic query-style label for the coverage report."""
    q = (query or "").strip()
    if not q:
        return "empty"
    sentences = [s for s in re.split(r"[.!?]+", q) if s.strip()]
    words = q.split()
    if len(sentences) >= 2 and len(words) >= 12:
        return "multi_sentence"
    if q.endswith("?"):
        return "question"
    if len(words) <= 3:
        return "keyword"
    return "statement"


def stable_key(row: Dict[str, Any]) -> str:
    """Deterministic per-row key (no RNG) for reproducible ordering/sampling."""
    raw = "\x1f".join(str(row.get(k, "")) for k in ("source_id", "query", "document"))
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()


def validate_row(row: Any, idx: int) -> List[str]:
    """Schema errors for one row (does not include license/leakage — those are separate gates)."""
    errs: List[str] = []
    if not isinstance(row, dict):
        return [f"row[{idx}]: not a JSON object"]
    for k in REQUIRED_STR_FIELDS:
        v = row.get(k)
        if not isinstance(v, str) or not v.strip():
            errs.append(f"row[{idx}] ({row.get('source_id', '?')}): '{k}' must be a non-empty string")
    dom = row.get("domain")
    if isinstance(dom, str) and dom not in TRAIN_DOMAINS:
        errs.append(f"row[{idx}] ({row.get('source_id', '?')}): unknown domain '{dom}' "
                    f"(allowed: {', '.join(TRAIN_DOMAINS)})")
    if not isinstance(row.get("synthetic_query"), bool):
        errs.append(f"row[{idx}] ({row.get('source_id', '?')}): 'synthetic_query' must be a bool")
    for k in ("eval_only", "public_benchmark"):
        if k in row and not isinstance(row[k], bool):
            errs.append(f"row[{idx}] ({row.get('source_id', '?')}): '{k}' must be a bool if present")
    return errs


def leakage_reason(row: Dict[str, Any]) -> str | None:
    """Why a row is public-benchmark/eval leakage (must NOT train), or None if clean."""
    if row.get("public_benchmark") is True:
        return "public_benchmark=true"
    if row.get("eval_only") is True:
        return "eval_only=true"
    for field in ("source_id", "source_url", "domain"):
        v = row.get(field)
        if isinstance(v, str) and is_public_benchmark(v):
            return f"{field} references a public benchmark ('{v}')"
    return None


def _balanced_sample(by_domain: Dict[str, List[Dict[str, Any]]], target: int) -> List[Dict[str, Any]]:
    """Round-robin over sorted domains; rows pre-sorted by stable key. Fully deterministic.

    Each cycle takes one row from each non-exhausted domain in domain-sorted order. When non-FAQ
    domains are exhausted the remaining budget is filled from whatever domains still have rows
    (typically FAQ) — that over-FAQ result is then caught by the share gate, never shipped."""
    domains = sorted(by_domain)
    pointers = {d: 0 for d in domains}
    selected: List[Dict[str, Any]] = []
    while len(selected) < target and any(pointers[d] < len(by_domain[d]) for d in domains):
        for d in domains:
            if len(selected) >= target:
                break
            if pointers[d] < len(by_domain[d]):
                selected.append(by_domain[d][pointers[d]])
                pointers[d] += 1
    return selected


def _counts(rows: List[Dict[str, Any]], key) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        out[str(key(r))] = out.get(str(key(r)), 0) + 1
    return dict(sorted(out.items()))


def mix(rows: List[Dict[str, Any]], *, target_count: int,
        max_faq_share: float, min_nonfaq_share: float) -> Dict[str, Any]:
    """Validate, deterministically balance-sample, gate shares, and build the coverage report.

    Returns a report dict with ``status`` ("pass"/"fail"), ``errors`` (fatal problems), the
    selected rows under ``selected``, and the coverage sections."""
    errors: List[str] = []

    # 1. schema
    schema_errors: List[str] = []
    for i, r in enumerate(rows):
        schema_errors += validate_row(r, i)
    valid = [r for i, r in enumerate(rows)
             if isinstance(r, dict) and not validate_row(r, i)]

    # 2. license (hard fail) — checked on schema-valid rows
    bad_license = sorted({normalize_license(r.get("license")) or "(empty)"
                          for r in valid if is_unknown_license(r.get("license"))})
    # 3. leakage (hard fail)
    leaks = [(r.get("source_id", "?"), leakage_reason(r)) for r in valid if leakage_reason(r)]

    if schema_errors:
        errors.append(f"{len(schema_errors)} schema error(s); first: {schema_errors[0]}")
    if bad_license:
        errors.append(f"unknown/unpermitted license(s): {bad_license}")
    if leaks:
        errors.append(f"public-benchmark/eval leakage in {len(leaks)} row(s); "
                      f"first: {leaks[0][0]} ({leaks[0][1]})")

    # Only schema-valid, licensed, non-leaking rows are eligible to train.
    eligible = [r for r in valid
                if not is_unknown_license(r.get("license")) and leakage_reason(r) is None]

    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for r in eligible:
        by_domain.setdefault(r["domain"], []).append(r)
    for d in by_domain:
        by_domain[d].sort(key=lambda r: (stable_key(r), str(r.get("source_id"))))

    selected = _balanced_sample(by_domain, target_count)

    n = len(selected)
    faq = sum(1 for r in selected if r["domain"] == FAQ_DOMAIN)
    faq_share = (faq / n) if n else 0.0
    nonfaq_share = 1.0 - faq_share if n else 0.0

    if n == 0:
        errors.append("no eligible rows to sample (after schema/license/leakage gates)")
    else:
        if faq_share > max_faq_share + 1e-9:
            errors.append(f"FAQ share {faq_share:.3f} exceeds --max-faq-share {max_faq_share} "
                          f"(add more non-FAQ data; v5 must not be FAQ-only)")
        if nonfaq_share < min_nonfaq_share - 1e-9:
            errors.append(f"non-FAQ share {nonfaq_share:.3f} below --min-nonfaq-share "
                          f"{min_nonfaq_share} (insufficient non-FAQ coverage)")

    synth = sum(1 for r in selected if r.get("synthetic_query") is True)
    examples: Dict[str, List[Dict[str, str]]] = {}
    for r in selected:
        bucket = examples.setdefault(r["domain"], [])
        if len(bucket) < 3:
            bucket.append({"source_id": str(r.get("source_id")), "query": r["query"],
                           "document_preview": r["document"][:160]})

    return {
        "status": "fail" if errors else "pass",
        "errors": errors,
        "params": {"target_count": target_count, "max_faq_share": max_faq_share,
                   "min_nonfaq_share": min_nonfaq_share},
        "input_rows": len(rows), "schema_valid_rows": len(valid),
        "eligible_rows": len(eligible), "selected_rows": n,
        "rows_by_domain": _counts(selected, lambda r: r["domain"]),
        "rows_by_source": _counts(selected, lambda r: r.get("source_id")),
        "rows_by_license": _counts(selected, lambda r: normalize_license(r.get("license"))),
        "faq_share": round(faq_share, 4), "nonfaq_share": round(nonfaq_share, 4),
        "synthetic_share": round(synth / n, 4) if n else 0.0,
        "real_share": round((n - synth) / n, 4) if n else 0.0,
        "query_style_distribution": _counts(selected, lambda r: classify_query_style(r["query"])),
        "examples_per_domain": examples,
        "available_by_domain": {d: len(v) for d, v in sorted(by_domain.items())},
        "selected": selected,
    }
