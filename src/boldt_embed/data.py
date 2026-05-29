"""Training-data schema, validation, license, and benchmark-leakage checks (pure stdlib).

A training record is one JSON object per line (JSONL):

    {
      "query": "...",            # required
      "positive": "...",         # required
      "negatives": ["...", ...], # optional; if absent, in-batch negatives are used
      "neg_types": ["compound"], # optional; German hard-negative categories
      "source": "synthetic-de",  # required provenance tag
      "license": "synthetic",    # required; must be in ALLOWED_LICENSES
      "lang": "de"               # optional, default "de"
    }

This module enforces ADR-004 (data licensing) and ADR-005 (leakage control).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .textutil import normalize, tokenize

# Permissive licenses we accept for training data (lowercased). "synthetic" denotes
# data we generated ourselves; its generator/prompt must still be versioned.
ALLOWED_LICENSES: Set[str] = {
    "cc0-1.0",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "apache-2.0",
    "mit",
    "public-domain",
    "synthetic",
}

# German hard-negative families (plus generic tags).
NEG_TYPES: Set[str] = {
    "compound",
    "negation",
    "legal_ref",
    "dates_numbers",
    "regional_variant",
    "entity_disambiguation",
    "lexical",
    "random",
}

REQUIRED_STR_FIELDS = ("query", "positive", "source", "license")


@dataclass
class DataReport:
    num_records: int
    num_with_negatives: int
    licenses: List[str]
    neg_type_counts: Dict[str, int]
    errors: List[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def load_jsonl(path: str | Path) -> List[dict]:
    records: List[dict] = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{i}: invalid JSON: {exc}") from exc
    return records


def validate_record(rec: object, idx: int) -> List[str]:
    errs: List[str] = []
    if not isinstance(rec, dict):
        return [f"record {idx}: not a JSON object"]

    for field in REQUIRED_STR_FIELDS:
        value = rec.get(field)
        if not isinstance(value, str) or not value.strip():
            errs.append(f"record {idx}: missing/empty required field '{field}'")

    license_value = rec.get("license")
    if isinstance(license_value, str) and license_value.strip():
        if license_value.strip().lower() not in ALLOWED_LICENSES:
            errs.append(f"record {idx}: disallowed license '{license_value}'")

    lang = rec.get("lang", "de")
    if not isinstance(lang, str) or not lang.strip():
        errs.append(f"record {idx}: 'lang' must be a non-empty string")

    negatives = rec.get("negatives")
    if negatives is not None:
        if not isinstance(negatives, list) or not all(
            isinstance(n, str) and n.strip() for n in negatives
        ):
            errs.append(f"record {idx}: 'negatives' must be a list of non-empty strings")
        else:
            positive = rec.get("positive")
            if isinstance(positive, str):
                pos_norm = normalize(positive)
                if any(normalize(n) == pos_norm for n in negatives):
                    errs.append(f"record {idx}: a negative is identical to the positive")

    neg_types = rec.get("neg_types")
    if neg_types is not None:
        if not isinstance(neg_types, list) or not all(isinstance(t, str) for t in neg_types):
            errs.append(f"record {idx}: 'neg_types' must be a list of strings")
        else:
            for t in neg_types:
                if t not in NEG_TYPES:
                    errs.append(f"record {idx}: unknown neg_type '{t}'")
    return errs


def validate_dataset(records: Sequence[dict]) -> DataReport:
    errors: List[str] = []
    licenses: Set[str] = set()
    neg_type_counts: Dict[str, int] = {}
    num_with_negatives = 0

    for idx, rec in enumerate(records):
        errors.extend(validate_record(rec, idx))
        if isinstance(rec, dict):
            lic = rec.get("license")
            if isinstance(lic, str) and lic.strip():
                licenses.add(lic.strip().lower())
            negs = rec.get("negatives")
            if isinstance(negs, list) and negs:
                num_with_negatives += 1
            for t in rec.get("neg_types", []) or []:
                if isinstance(t, str):
                    neg_type_counts[t] = neg_type_counts.get(t, 0) + 1

    return DataReport(
        num_records=len(records),
        num_with_negatives=num_with_negatives,
        licenses=sorted(licenses),
        neg_type_counts=dict(sorted(neg_type_counts.items())),
        errors=errors,
    )


def check_licenses(records: Sequence[dict]) -> List[str]:
    """Return the sorted list of disallowed license strings present (empty == OK)."""
    bad: Set[str] = set()
    for rec in records:
        lic = rec.get("license") if isinstance(rec, dict) else None
        if isinstance(lic, str) and lic.strip() and lic.strip().lower() not in ALLOWED_LICENSES:
            bad.add(lic.strip())
    return sorted(bad)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def find_leakage(
    records: Sequence[dict],
    eval_texts: Iterable[str],
    threshold: float = 0.9,
) -> List[dict]:
    """Detect training records that leak evaluation text.

    Flags exact matches (after normalization) and near-duplicates (token Jaccard >=
    threshold) between a record's ``query``/``positive`` and any evaluation text.
    """
    eval_list = [t for t in eval_texts if isinstance(t, str) and t.strip()]
    eval_by_norm = {normalize(t): t for t in eval_list}
    eval_sigs = [(t, set(tokenize(t))) for t in eval_list]

    hits: List[dict] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        for field in ("positive", "query"):
            text = rec.get(field)
            if not isinstance(text, str) or not text.strip():
                continue
            norm = normalize(text)
            if norm in eval_by_norm:
                hits.append(
                    {"record": i, "field": field, "kind": "exact",
                     "score": 1.0, "eval_text": eval_by_norm[norm]}
                )
                continue
            sig = set(tokenize(text))
            best_score, best_text = 0.0, None
            for cand_text, cand_sig in eval_sigs:
                score = _jaccard(sig, cand_sig)
                if score > best_score:
                    best_score, best_text = score, cand_text
            if best_score >= threshold:
                hits.append(
                    {"record": i, "field": field, "kind": "near_dup",
                     "score": round(best_score, 4), "eval_text": best_text}
                )
    return hits


# ------------------------------------------------------------------------------ PII
# Conservative German-aware PII patterns (ADR-004 safety dimension). Tuned to avoid
# flagging legal references like "§ 543" or plain years.
_PII_PATTERNS: Dict[str, "re.Pattern[str]"] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}"),
    "iban_de": re.compile(r"\bDE\d{2}[ ]?(?:\d{4}[ ]?){4}\d{2}\b"),
    "phone": re.compile(r"(?<![\w.])(?:\+49|0049|0)[\d][\d /()-]{6,}\d"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def find_pii(text: str) -> List[Tuple[str, str]]:
    """Return [(kind, matched_text)] of likely PII in a single string."""
    hits: List[Tuple[str, str]] = []
    if not isinstance(text, str):
        return hits
    for kind, pat in _PII_PATTERNS.items():
        for m in pat.finditer(text):
            hits.append((kind, m.group(0)))
    return hits


def scan_pii(records: Sequence[dict]) -> List[dict]:
    """Scan query/positive/negatives of every record for PII. Empty list == clean."""
    out: List[dict] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        fields = [("query", rec.get("query")), ("positive", rec.get("positive"))]
        for j, neg in enumerate(rec.get("negatives", []) or []):
            fields.append((f"negatives[{j}]", neg))
        for field, value in fields:
            for kind, match in find_pii(value if isinstance(value, str) else ""):
                out.append({"record": i, "field": field, "kind": kind, "match": match})
    return out
