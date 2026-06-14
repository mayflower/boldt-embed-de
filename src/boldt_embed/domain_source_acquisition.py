"""v3 real-domain source acquisition (pure stdlib, fail-closed).

v2's lesson: templated synthetic queries over Wikipedia did NOT generalize to admin/FAQ/legal —
the teacher rejected them. v3 sources real, licensed domain corpora. This module validates a v3
source manifest and materializes LOCAL drops, with hard fail-closed rules:

- a source may train (``allowed_for_training=true``) ONLY if ``license_verified=true`` AND it is
  not a public benchmark, not eval-only, carries no eval-overlap risk, and its license string is
  not an uncertain placeholder. If a license is uncertain, ``allowed_for_training`` MUST be false
  until a human verifies it and flips the flag.
- synthetic data is allowed only as ``supplemental=true`` and is NOT counted toward the
  real-domain (``*_real``) coverage targets.

No network here except the explicit ``download-hf`` mode (lazy ``datasets`` import); ``dry-run``
and ``materialize-local`` never touch the network.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SOURCE_TYPES = ("local_jsonl", "local_corpus_jsonl", "hf_dataset", "url_manifest")
TRAINING_DOMAINS = ("faq_real", "admin_real", "legal_adjacency_real_no_eval_overlap",
                    "web", "wiki_non_eval", "german_stress", "cross_lingual_de_en")
REAL_REQUIRED_DOMAINS = ("faq_real", "admin_real", "legal_adjacency_real_no_eval_overlap")
UNCERTAIN_LICENSE_MARKERS = ("uncertain", "unknown", "verify", "tbd", "todo", "?")

_BOOL_FIELDS = ("license_verified", "allowed_for_training", "eval_only", "public_benchmark",
                "contains_eval_overlap_risk", "requires_attribution")


def license_is_uncertain(license_str: str) -> bool:
    low = (license_str or "").strip().lower()
    return (not low) or any(m in low for m in UNCERTAIN_LICENSE_MARKERS)


@dataclass
class V3SourceEntry:
    source_id: str
    display_name: str
    domain: str
    source_type: str
    license: str
    license_url: str
    license_verified: bool
    allowed_for_training: bool
    eval_only: bool
    public_benchmark: bool
    contains_eval_overlap_risk: bool
    requires_attribution: bool
    notes: str
    loader: Dict[str, Any]
    expected_fields: Dict[str, Any] = field(default_factory=dict)
    supplemental: bool = False          # synthetic/supplemental: never counts as real-domain
    raw: Dict[str, Any] = field(default_factory=dict)

    def loader_path(self) -> Optional[str]:
        return self.loader.get("path") or self.loader.get("path_or_id")


def _entry(d: Dict[str, Any]) -> V3SourceEntry:
    return V3SourceEntry(
        source_id=d["source_id"], display_name=d.get("display_name", d["source_id"]),
        domain=d["domain"], source_type=d["source_type"], license=d.get("license", ""),
        license_url=d.get("license_url", ""), license_verified=bool(d.get("license_verified", False)),
        allowed_for_training=bool(d.get("allowed_for_training", False)),
        eval_only=bool(d.get("eval_only", False)), public_benchmark=bool(d.get("public_benchmark", False)),
        contains_eval_overlap_risk=bool(d.get("contains_eval_overlap_risk", False)),
        requires_attribution=bool(d.get("requires_attribution", False)),
        notes=d.get("notes", ""), loader=dict(d.get("loader") or {}),
        expected_fields=dict(d.get("expected_fields") or {}),
        supplemental=bool(d.get("supplemental", False)), raw=d)


# --------------------------------------------------------------------------- validate
def validate_v3_source(d: Dict[str, Any]) -> List[str]:
    e: List[str] = []
    sid = d.get("source_id")
    if not isinstance(sid, str) or not sid.strip():
        return ["source missing 'source_id'"]
    for f in ("display_name", "domain", "source_type", "license"):
        if not isinstance(d.get(f), str) or not d[f].strip():
            e.append(f"{sid}: '{f}' must be a non-empty string")
    for f in _BOOL_FIELDS:
        if f in d and not isinstance(d[f], bool):
            e.append(f"{sid}: '{f}' must be a bool")
    if "supplemental" in d and not isinstance(d["supplemental"], bool):
        e.append(f"{sid}: 'supplemental' must be a bool")
    if d.get("source_type") not in SOURCE_TYPES:
        e.append(f"{sid}: source_type must be one of {SOURCE_TYPES}")
    if d.get("domain") not in TRAINING_DOMAINS:
        e.append(f"{sid}: domain '{d.get('domain')}' not in {TRAINING_DOMAINS}")
    if not isinstance(d.get("loader"), dict):
        e.append(f"{sid}: 'loader' must be an object")

    allowed = bool(d.get("allowed_for_training", False))
    if allowed:
        # FAIL-CLOSED: training requires a verified, concrete, non-benchmark, overlap-free source.
        if not bool(d.get("license_verified", False)):
            e.append(f"{sid}: allowed_for_training=true requires license_verified=true "
                     "(uncertain license -> allowed_for_training must be false)")
        if license_is_uncertain(d.get("license", "")):
            e.append(f"{sid}: allowed_for_training=true but license is uncertain/empty")
        if bool(d.get("public_benchmark", False)):
            e.append(f"{sid}: a public_benchmark source cannot be a training source")
        if bool(d.get("eval_only", False)):
            e.append(f"{sid}: an eval_only source cannot be a training source")
        if bool(d.get("contains_eval_overlap_risk", False)):
            e.append(f"{sid}: contains_eval_overlap_risk=true -> run the full leakage scan and "
                     "clear the flag before allowing training")
    if bool(d.get("license_verified", False)) and license_is_uncertain(d.get("license", "")):
        e.append(f"{sid}: license_verified=true is inconsistent with an uncertain license string")
    return e


def validate_v3_manifest(d: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    sources = d.get("sources")
    if not isinstance(sources, list) or not sources:
        return ["'sources' must be a non-empty list"]
    seen = set()
    for s in sources:
        errors += validate_v3_source(s)
        sid = s.get("source_id")
        if sid in seen:
            errors.append(f"duplicate source_id: {sid}")
        seen.add(sid)
    return errors


def load_v3_manifest(path: str | Path) -> List[V3SourceEntry]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_v3_manifest(d)
    if errors:
        raise ValueError("invalid v3 source manifest: " + "; ".join(errors))
    return [_entry(s) for s in d["sources"]]


# ----------------------------------------------------------------------- row validation
def validate_local_jsonl_row(row: Dict[str, Any]) -> List[str]:
    """A local data row needs an id, text (or query+document), source_id and license.
    url/title optional."""
    e: List[str] = []
    if not isinstance(row, dict):
        return ["row is not an object"]
    if not (row.get("id") or row.get("doc_id") or row.get("query_id")):
        e.append("missing id (id/doc_id/query_id)")
    has_text = isinstance(row.get("text"), str) and row["text"].strip()
    has_pair = (isinstance(row.get("query"), str) and row["query"].strip()
                and isinstance(row.get("document"), str) and row["document"].strip())
    if not (has_text or has_pair):
        e.append("missing 'text' or ('query' and 'document')")
    if not (isinstance(row.get("source_id"), str) and row["source_id"].strip()):
        e.append("missing 'source_id'")
    if not (isinstance(row.get("license"), str) and row["license"].strip()):
        e.append("missing 'license'")
    return e


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


# ---------------------------------------------------------------------------- acquire
def block_reason(e: V3SourceEntry, mode: str) -> Optional[str]:
    """Why a source is NOT materialized in this mode (None = materializable)."""
    if not e.allowed_for_training:
        if not e.license_verified:
            return "blocked: license not verified (allowed_for_training=false)"
        if e.public_benchmark:
            return "blocked: public_benchmark (eval-only, never train)"
        if e.eval_only:
            return "blocked: eval_only"
        return "blocked: allowed_for_training=false"
    if e.source_type in ("local_jsonl", "local_corpus_jsonl"):
        return None if mode in ("materialize-local", "dry-run") else f"skipped in mode={mode}"
    if e.source_type == "hf_dataset":
        return None if mode == "download-hf" else f"hf_dataset not downloaded in mode={mode}"
    if e.source_type == "url_manifest":
        return "url_manifest: metadata only (no scraping)"
    return f"unhandled source_type {e.source_type}"


def acquire(entries: List[V3SourceEntry], output_dir: str, mode: str,
            fail_on_unverified_license: bool = False) -> Dict[str, Any]:
    """Plan/execute acquisition and return a summary. dry-run plans only; materialize-local
    reads+validates+writes local drops; download-hf additionally pulls hf_dataset sources."""
    out = Path(output_dir)
    rows_by_source: Dict[str, int] = {}
    rows_by_domain: Dict[str, int] = {}
    rows_by_license: Dict[str, int] = {}
    materialized: List[Dict[str, Any]] = []
    blocked: List[Dict[str, str]] = []
    errors: List[str] = []

    for e in entries:
        reason = block_reason(e, mode)
        if reason is not None:
            blocked.append({"source_id": e.source_id, "domain": e.domain, "reason": reason})
            continue
        path = e.loader_path()
        if e.source_type in ("local_jsonl", "local_corpus_jsonl"):
            if not path or not Path(path).exists():
                blocked.append({"source_id": e.source_id, "domain": e.domain,
                                "reason": f"local data not present: {path}"})
                continue
            if mode == "dry-run":
                materialized.append({"source_id": e.source_id, "domain": e.domain,
                                     "planned_input": path, "rows": None})
                continue
            kept = []
            bad = 0
            for r in _read_jsonl(path):
                if validate_local_jsonl_row(r):
                    bad += 1
                    continue
                r.setdefault("source_id", e.source_id)
                r.setdefault("license", e.license)
                if e.source_type == "local_corpus_jsonl":
                    r.setdefault("generated", False)  # raw documents; query-gen is marked later
                kept.append(r)
            if bad:
                errors.append(f"{e.source_id}: {bad} invalid rows dropped")
            out.mkdir(parents=True, exist_ok=True)
            dst = out / f"{e.source_id}.jsonl"
            with dst.open("w", encoding="utf-8") as f:
                for r in kept:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            rows_by_source[e.source_id] = len(kept)
            rows_by_domain[e.domain] = rows_by_domain.get(e.domain, 0) + len(kept)
            rows_by_license[e.license] = rows_by_license.get(e.license, 0) + len(kept)
            materialized.append({"source_id": e.source_id, "domain": e.domain,
                                 "rows": len(kept), "path": str(dst),
                                 "supplemental": e.supplemental})
        elif e.source_type == "hf_dataset":   # mode == download-hf
            materialized.append({"source_id": e.source_id, "domain": e.domain,
                                 "rows": None, "note": "download-hf not run in this environment"})

    # real-domain coverage counts NON-supplemental materialized sources only.
    real_cov = {d: 0 for d in REAL_REQUIRED_DOMAINS}
    for m in materialized:
        if m["domain"] in real_cov and not m.get("supplemental"):
            real_cov[m["domain"]] += 1
    supplemental_sources = [e.source_id for e in entries if e.supplemental]
    unverified = [e.source_id for e in entries if not e.license_verified]

    status = "ok"
    if errors:
        status = "fail"
    if fail_on_unverified_license and unverified:
        status = "fail"
        errors.append("fail-on-unverified-license: unverified sources present: "
                      + ", ".join(unverified))

    return {
        "status": status, "mode": mode, "output_dir": str(out),
        "n_sources": len(entries),
        "rows_by_source": dict(sorted(rows_by_source.items())),
        "rows_by_domain": dict(sorted(rows_by_domain.items())),
        "rows_by_license": dict(sorted(rows_by_license.items())),
        "materialized": materialized,
        "blocked": blocked,
        "real_domain_coverage": real_cov,
        "real_domains_missing": [d for d, n in real_cov.items() if n == 0],
        "supplemental_sources": supplemental_sources,
        "unverified_sources": unverified,
        "errors": errors,
    }
