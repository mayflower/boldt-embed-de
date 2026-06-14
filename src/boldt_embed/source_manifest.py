"""v2 data-source manifest: load + validate (pure stdlib, no ML, no network).

The manifest (`configs/data_sources_v2.json`) is the auditable gate for what may enter v2
training. It **fails closed**: a source is training-allowed ONLY if it has a concrete license,
is not a public benchmark, and is not eval-only. Public benchmark / eval-only / uncertain-
license sources are blocked from training by validation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

SOURCE_TYPES = {"local_jsonl", "hf_dataset", "synthetic", "derived"}
LOADER_KINDS = {"jsonl", "hf"}

# Domains usable for TRAINING (must match configs/experiments/v2_generalization.json).
TRAINING_DOMAINS = {
    "web", "faq", "admin", "legal_adjacency_no_eval_overlap",
    "wiki_non_eval", "german_stress", "cross_lingual_de_en",
}
# Content tags allowed for eval-only/benchmark sources (never trained on).
EVAL_DOMAINS = {"qa_wiki", "legal", "sts", "clustering"}
KNOWN_DOMAINS = TRAINING_DOMAINS | EVAL_DOMAINS

# License strings that mean "not actually known" -> cannot be training-allowed.
UNCERTAIN_LICENSE_MARKERS = ("uncertain", "unknown", "verify", "tbd", "todo", "?")


@dataclass
class SourceEntry:
    source_id: str
    display_name: str
    source_type: str
    domain: str
    license: str
    allowed_for_training: bool
    public_benchmark: bool
    eval_only: bool
    notes: str
    loader: Dict[str, Any]
    expected_fields: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


def _license_is_uncertain(lic: str) -> bool:
    low = (lic or "").lower()
    return (not low.strip()) or any(m in low for m in UNCERTAIN_LICENSE_MARKERS)


def validate_source_entry(d: Any, idx: int = 0) -> List[str]:
    """Return problems with one source entry (never raises)."""
    p = f"source[{idx}]"
    if not isinstance(d, dict):
        return [f"{p} must be an object"]
    sid = d.get("source_id")
    p = f"source '{sid}'" if isinstance(sid, str) and sid else p
    errors: List[str] = []
    if not isinstance(sid, str) or not sid.strip():
        errors.append(f"{p}: missing/empty source_id")
    if d.get("source_type") not in SOURCE_TYPES:
        errors.append(f"{p}: source_type must be one of {sorted(SOURCE_TYPES)}")
    domain = d.get("domain")
    if domain not in KNOWN_DOMAINS:
        errors.append(f"{p}: unknown domain '{domain}' (allowed: {sorted(KNOWN_DOMAINS)})")
    lic = d.get("license")
    if not isinstance(lic, str) or not lic.strip():
        errors.append(f"{p}: missing license")
    for b in ("allowed_for_training", "public_benchmark", "eval_only"):
        if not isinstance(d.get(b), bool):
            errors.append(f"{p}: '{b}' must be a bool")
    loader = d.get("loader")
    if not isinstance(loader, dict) or loader.get("kind") not in LOADER_KINDS \
            or not str(loader.get("path_or_id", "")).strip():
        errors.append(f"{p}: loader must have kind in {sorted(LOADER_KINDS)} + a path_or_id")

    # --- fail-closed training-eligibility rules ---
    allowed = d.get("allowed_for_training") is True
    if allowed and d.get("public_benchmark") is True:
        errors.append(f"{p}: public_benchmark sources MUST NOT be allowed_for_training")
    if allowed and d.get("eval_only") is True:
        errors.append(f"{p}: eval_only sources MUST NOT be allowed_for_training")
    if allowed and isinstance(lic, str) and _license_is_uncertain(lic):
        errors.append(f"{p}: uncertain license '{lic}' -> allowed_for_training must be false")
    if allowed and isinstance(domain, str) and domain not in TRAINING_DOMAINS:
        errors.append(f"{p}: training-allowed source must use a training domain "
                      f"(got '{domain}'; not in {sorted(TRAINING_DOMAINS)})")
    return errors


def validate_source_manifest(d: Dict[str, Any]) -> List[str]:
    sources = d.get("sources")
    if not isinstance(sources, list) or not sources:
        return ["'sources' must be a non-empty list"]
    errors: List[str] = []
    seen = set()
    for i, s in enumerate(sources):
        errors += validate_source_entry(s, i)
        sid = s.get("source_id") if isinstance(s, dict) else None
        if sid in seen:
            errors.append(f"duplicate source_id '{sid}'")
        seen.add(sid)
    return errors


def _entry(d: Dict[str, Any]) -> SourceEntry:
    return SourceEntry(
        source_id=d["source_id"], display_name=d.get("display_name", d["source_id"]),
        source_type=d["source_type"], domain=d["domain"], license=d["license"],
        allowed_for_training=bool(d["allowed_for_training"]),
        public_benchmark=bool(d["public_benchmark"]), eval_only=bool(d["eval_only"]),
        notes=d.get("notes", ""), loader=dict(d["loader"]),
        expected_fields=dict(d.get("expected_fields") or {}), raw=d)


def load_source_manifest(path: str | Path) -> List[SourceEntry]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_source_manifest(d)
    if errors:
        raise ValueError("invalid source manifest: " + "; ".join(errors))
    return [_entry(s) for s in d["sources"]]


def training_sources(entries: List[SourceEntry]) -> List[SourceEntry]:
    """The subset safe to train on (already validated as not eval/benchmark/uncertain)."""
    return [e for e in entries if e.allowed_for_training]


def license_origin_for(license_str: str | None) -> str:
    """Where a row's license came from: a manifest source carrying a concrete license
    (``manifest``) vs a synthetic source whose license is inherited from its seed passages
    (``inherited``, e.g. the manifest marker ``synthetic-inherits-source``)."""
    return "inherited" if "inherit" in (license_str or "").lower() else "manifest"


def candidate_provenance(entry: SourceEntry, row: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Full, explicit provenance for a candidate row built from a manifest source.

    The manifest is authoritative for ``allowed_for_training`` and the *kind* of license.
    For inherited (synthetic) sources we prefer a CONCRETE inherited license carried on the
    raw row (set by the generator from the seed passage) over the bare ``...-inherits-source``
    marker, and record ``inherited_from_source_id`` when the row knows its seed.

    Always returns: source_id, source, license, license_origin, allowed_for_training. Adds
    license_url and inherited_from_source_id only when known. (domain is set by the caller.)
    """
    row = row or {}
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    origin = license_origin_for(entry.license)
    if origin == "inherited":
        row_lic = row.get("license")
        concrete = (row_lic if isinstance(row_lic, str) and row_lic.strip()
                    and "inherit" not in row_lic.lower() else None)
        license_ = concrete or entry.license
    else:
        license_ = entry.license
    prov: Dict[str, Any] = {
        "source_id": entry.source_id,
        "source": entry.source_id,
        "license": license_,
        "license_origin": origin,
        "allowed_for_training": bool(entry.allowed_for_training),
    }
    url = row.get("license_url") or meta.get("license_url") or entry.raw.get("license_url")
    if url:
        prov["license_url"] = url
    if origin == "inherited":
        ifrom = (row.get("inherited_from_source_id") or meta.get("inherited_from_source_id")
                 or meta.get("source_passage_source") or row.get("source_passage_source"))
        if ifrom:
            prov["inherited_from_source_id"] = ifrom
    return prov


def render_markdown(entries: List[SourceEntry]) -> str:
    lines = ["# v2 data sources", "",
             "| source_id | domain | license | train? | eval_only | public_bench |",
             "|---|---|---|:--:|:--:|:--:|"]
    for e in entries:
        lines.append(f"| {e.source_id} | {e.domain} | {e.license} | "
                     f"{'✅' if e.allowed_for_training else '—'} | "
                     f"{'✅' if e.eval_only else '—'} | {'✅' if e.public_benchmark else '—'} |")
    n_train = len(training_sources(entries))
    lines += ["", f"{len(entries)} sources; {n_train} training-allowed."]
    return "\n".join(lines) + "\n"
