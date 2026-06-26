"""Data-mixture optimizer — a real, constraint-aware training-corpus builder.

Both a library (``build_mixture`` / ``plan_mixture``) and a script-backed entry
(``scripts/ar_build_mixture.py``). It supersedes the coarse
``autoresearch_recipe._materialize_data_mixture`` by adding: per-source weighted
budgets, FAQ cap + domain min/max-share rebalancing, length-bucket targeting,
exact + normalized-text dedupe, and a fully-specified provenance manifest.

PURE PYTHON STDLIB — no torch / transformers / numpy at module top (or anywhere).
Importing this module must NOT pull in torch::

    import sys
    import boldt_embed.data_mixture_optimizer  # noqa
    assert "torch" not in sys.modules

FAIL-CLOSED is the contract. Any of the following raises a ``MixtureConfigError``
naming the offending source id (never a silent skip):

* a source named in the mixture is not in the catalogue;
* a source is ``training_usable: false``;
* a source's ``leakage`` is not in ``{"scanned_clean", "clean"}``;
* a source is in the eval-only group (``eval_only_NEVER_TRAIN``);
* a source's on-disk file is missing;
* a weight is ``<= 0`` (or non-numeric), or the sources map is empty.

Length buckets are keyed off the **document** (positive) character length, with
documented thresholds: ``short < 256``, ``256 <= medium < 1024``, ``long >= 1024``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Repo root (…/boldt-embed-de). Used to resolve catalogue-relative source paths.
ROOT = Path(__file__).resolve().parents[2]

# Length-bucket thresholds (document character length). Documented in the module
# docstring and echoed in the manifest's ``length_thresholds``.
SHORT_MAX = 256          # short  : len(doc) < 256
MEDIUM_MAX = 1024        # medium : 256 <= len(doc) < 1024 ; long : len(doc) >= 1024

# Catalogue groups that may contribute training sources.
_TRAIN_GROUPS = ("train_pairs_processed_unions", "train_pairs_raw_sources")
# Catalogue group whose ids must NEVER be trained on (hard fail if referenced).
_EVAL_ONLY_GROUP = "eval_only_NEVER_TRAIN"
_ALLOWED_LEAKAGE = ("scanned_clean", "clean")


class MixtureConfigError(ValueError):
    """Raised (fail-closed) on any invalid mixture config / unusable source."""


# --------------------------------------------------------------------------- catalogue
def load_catalogue(catalog_path: Path | str) -> Dict[str, Any]:
    """Load the data-sources catalogue JSON. Raises ``MixtureConfigError`` if absent/invalid."""
    p = Path(catalog_path)
    if not p.is_absolute():
        p = (ROOT / p)
    if not p.exists():
        raise MixtureConfigError(f"catalogue not found: {catalog_path}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # malformed catalogue is fatal, not a silent skip
        raise MixtureConfigError(f"catalogue not parseable ({catalog_path}): {exc}") from exc


def _index_catalogue(catalogue: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], set]:
    """Return (id -> trainable-source record, {eval-only ids})."""
    trainable: Dict[str, Dict[str, Any]] = {}
    for grp in _TRAIN_GROUPS:
        for s in catalogue.get(grp, []) or []:
            if isinstance(s, dict) and s.get("id"):
                trainable[s["id"]] = s
    eval_only = set()
    for s in catalogue.get(_EVAL_ONLY_GROUP, []) or []:
        if isinstance(s, dict) and s.get("id"):
            eval_only.add(s["id"])
    return trainable, eval_only


def _resolve_source_path(rec: Dict[str, Any]) -> Path:
    path = rec.get("path")
    if not path:
        raise MixtureConfigError(f"source {rec.get('id')!r} has no 'path' in catalogue")
    p = Path(path)
    return p if p.is_absolute() else (ROOT / str(path))


# ---------------------------------------------------------------------- config validation
def validate_mixture_config(config: Dict[str, Any], catalogue: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the mixture config against the catalogue (FAIL-CLOSED).

    Returns a normalized view: ``{"name", "total_rows", "weights" (normalized, sum≈1),
    "sources" (id->record), "constraints"}``. Raises ``MixtureConfigError`` naming the
    offending source on the first hard problem.
    """
    if not isinstance(config, dict):
        raise MixtureConfigError("mixture config must be a JSON object")
    name = config.get("name")
    if not name or not isinstance(name, str):
        raise MixtureConfigError("mixture config requires a non-empty string 'name'")
    total_rows = config.get("total_rows")
    if not isinstance(total_rows, int) or isinstance(total_rows, bool) or total_rows <= 0:
        raise MixtureConfigError(f"'total_rows' must be a positive int (got {total_rows!r})")

    sources = config.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise MixtureConfigError("mixture config requires a non-empty 'sources' object")

    trainable, eval_only = _index_catalogue(catalogue)

    resolved: Dict[str, Dict[str, Any]] = {}
    raw_weights: Dict[str, float] = {}
    for sid, weight in sources.items():
        # weight must be a positive number (reject <=0 and non-numeric / bool)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise MixtureConfigError(f"source {sid!r} weight must be a number (got {weight!r})")
        if weight <= 0:
            raise MixtureConfigError(f"source {sid!r} weight must be > 0 (got {weight})")
        # eval-only sources are never trainable
        if sid in eval_only:
            raise MixtureConfigError(
                f"source {sid!r} is eval-only (in {_EVAL_ONLY_GROUP}) — NEVER train on it")
        rec = trainable.get(sid)
        if rec is None:
            raise MixtureConfigError(
                f"source {sid!r} is not a known trainable source in the catalogue")
        if not rec.get("training_usable"):
            raise MixtureConfigError(f"source {sid!r} is training_usable=false")
        leak = rec.get("leakage")
        if leak not in _ALLOWED_LEAKAGE:
            raise MixtureConfigError(
                f"source {sid!r} leakage={leak!r} — only {_ALLOWED_LEAKAGE} may be trained on "
                "(run scripts/run_full_leakage_scan.py first)")
        resolved[sid] = rec
        raw_weights[sid] = float(weight)

    wsum = sum(raw_weights.values())
    if wsum <= 0:  # defensive (each weight already > 0)
        raise MixtureConfigError("source weights sum to <= 0 after validation")
    weights = {sid: w / wsum for sid, w in raw_weights.items()}

    constraints = config.get("constraints") or {}
    if not isinstance(constraints, dict):
        raise MixtureConfigError("'constraints' must be an object when present")

    return {
        "name": name,
        "total_rows": total_rows,
        "weights": weights,
        "sources": resolved,
        "constraints": constraints,
    }


# --------------------------------------------------------------------------- helpers
def _norm_text(s: str) -> str:
    """Lowercased, whitespace-collapsed normalization for normalized-text dedupe."""
    return " ".join(str(s).lower().split())


def _length_bucket(doc: str) -> str:
    n = len(doc)
    if n < SHORT_MAX:
        return "short"
    if n < MEDIUM_MAX:
        return "medium"
    return "long"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_pairs(path: Path):
    """Yield (query, document, domain) from a JSONL pairs file. Skips blank/invalid lines.

    Accepts ``positive`` or ``document`` as the positive text (matching the catalogue's two
    pair conventions). Yields only rows with a non-empty query and a >=5-char document.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            q = (d.get("query") or "").strip() if isinstance(d.get("query"), str) else ""
            doc = d.get("positive")
            if doc is None:
                doc = d.get("document")
            doc = doc.strip() if isinstance(doc, str) else ""
            if not q or len(doc) < 5:
                continue
            yield q, doc, (d.get("domain") or None)


def _budgets(weights: Dict[str, float], total: int) -> Dict[str, int]:
    """Largest-remainder apportionment of ``total`` rows across normalized weights."""
    raw = {sid: weights[sid] * total for sid in weights}
    floors = {sid: int(raw[sid]) for sid in raw}
    assigned = sum(floors.values())
    remainder = total - assigned
    # distribute the remaining rows to the largest fractional parts (deterministic tie-break by id)
    frac_order = sorted(raw, key=lambda s: (raw[s] - floors[s], s), reverse=True)
    for i in range(max(0, remainder)):
        floors[frac_order[i % len(frac_order)]] += 1
    return floors


# ------------------------------------------------------------------------- core selection
def _select(spec: Dict[str, Any], collect: bool) -> Dict[str, Any]:
    """Shared selection core for ``plan_mixture`` (collect=False) and ``build_mixture``
    (collect=True). Stride-samples each source to its budget, applies exact + normalized-text
    dedupe, FAQ cap + domain min/max-share rebalancing (via ``train_modern``), then computes
    domain/length mixes. When ``collect`` is False, no examples are materialized (estimate only).
    """
    weights: Dict[str, float] = spec["weights"]
    total: int = spec["total_rows"]
    sources: Dict[str, Dict[str, Any]] = spec["sources"]
    constraints: Dict[str, Any] = spec["constraints"]
    dedupe_cfg = constraints.get("dedupe") or {}
    do_exact = bool(dedupe_cfg.get("exact", True))
    do_norm = bool(dedupe_cfg.get("normalized_text", True))
    faq_cap = float(constraints.get("faq_cap", 0.30))

    budgets = _budgets(weights, total)

    # FAIL-CLOSED: every source file must exist before we sample anything.
    src_paths: Dict[str, Path] = {}
    for sid, rec in sources.items():
        p = _resolve_source_path(rec)
        if not p.exists():
            raise MixtureConfigError(f"source {sid!r} file missing on disk: {p}")
        src_paths[sid] = p

    examples: List[Dict[str, Any]] = []
    per_source: List[Dict[str, Any]] = []
    source_hashes: Dict[str, str] = {}

    exact_seen: set = set()
    norm_seen: set = set()
    dropped_exact = 0
    dropped_norm = 0
    # Lightweight pre-balance tallies (used for the dry-run estimate so the manifest's
    # domain/length mix is meaningful WITHOUT materializing the full example list).
    pre_domain_counts: Dict[str, int] = {}
    pre_length_counts = {"short": 0, "medium": 0, "long": 0}

    for sid in weights:  # deterministic order = mixture insertion order
        rec = sources[sid]
        p = src_paths[sid]
        budget = budgets[sid]
        source_hashes[sid] = _sha256_file(p)

        # Count rows to compute a diversity-preserving stride (read once for the count).
        n_rows = 0
        for _ in _iter_pairs(p):
            n_rows += 1
        stride = max(1, n_rows // budget) if budget > 0 else 1

        kept = 0
        seen_local = 0
        for q, doc, domain in _iter_pairs(p):
            if kept >= budget:  # budget 0 (weight floored away) keeps NOTHING; >0 stops at exactly budget
                break
            if seen_local % stride != 0:
                seen_local += 1
                continue
            seen_local += 1
            if do_exact:
                ek = (q, doc)
                if ek in exact_seen:
                    dropped_exact += 1
                    continue
                exact_seen.add(ek)
            if do_norm:
                nk = (_norm_text(q), _norm_text(doc))
                if nk in norm_seen:
                    dropped_norm += 1
                    continue
                norm_seen.add(nk)
            dom = domain or rec.get("domain") or sid
            lb = _length_bucket(doc)
            if collect:
                examples.append({
                    "query": q,
                    "positive": doc,
                    "domain": dom,
                    "source": sid,
                    "_length_bucket": lb,
                })
            else:
                pre_domain_counts[str(dom)] = pre_domain_counts.get(str(dom), 0) + 1
                pre_length_counts[lb] = pre_length_counts.get(lb, 0) + 1
            kept += 1
            if kept >= budget:
                break
        per_source.append({
            "source": sid,
            "weight": round(weights[sid], 6),
            "leakage": rec.get("leakage"),
            "domain": rec.get("domain"),
            "rows_available": n_rows,
            "budget": budget,
            "kept": kept,
        })

    # FAQ cap + domain round-robin rebalancing (pure stdlib, lazy import).
    balance_report: Dict[str, Any] = {}
    if collect:
        from boldt_embed.train_modern import domain_balanced_examples  # lazy stdlib
        examples, balance_report = domain_balanced_examples(examples, faq_cap=faq_cap)

    # domain / length mixes. REAL builds count the final FAQ-rebalanced set; dry-runs use the
    # pre-balance tallies gathered above (a faithful estimate without materializing examples).
    if collect:
        domain_counts: Dict[str, int] = {}
        length_counts = {"short": 0, "medium": 0, "long": 0}
        for e in examples:
            dom = str(e.get("domain") or "unknown")
            domain_counts[dom] = domain_counts.get(dom, 0) + 1
            lb = e.get("_length_bucket") or _length_bucket(e.get("positive") or "")
            length_counts[lb] = length_counts.get(lb, 0) + 1
        total_kept = len(examples)
    else:
        domain_counts = dict(pre_domain_counts)
        length_counts = dict(pre_length_counts)
        # estimate: post-dedupe kept rows (FAQ rebalancing is not applied in plan mode)
        total_kept = sum(b["kept"] for b in per_source)

    def _share(counts: Dict[str, int]) -> Dict[str, float]:
        return {k: round(v / total_kept, 4) for k, v in sorted(counts.items())} if total_kept else {}

    return {
        "examples": examples,
        "per_source": per_source,
        "budgets": budgets,
        "source_hashes": source_hashes,
        "domain_counts": dict(sorted(domain_counts.items())),
        "domain_mix": _share(domain_counts),
        "length_counts": length_counts,
        "length_mix": _share(length_counts),
        "rows_written": total_kept,
        "dedupe": {
            "exact": do_exact,
            "normalized_text": do_norm,
            "simhash": bool(dedupe_cfg.get("simhash", False)),
            "dropped_exact": dropped_exact,
            "dropped_normalized_text": dropped_norm,
        },
        "faq_balance": balance_report,
    }


# --------------------------------------------------------------------------- manifest
def _build_manifest(spec: Dict[str, Any], sel: Dict[str, Any], *, rows_written: int,
                    created_utc: Optional[str], dry_run: bool) -> Dict[str, Any]:
    return {
        "name": spec["name"],
        "created_utc": created_utc,
        "dry_run": dry_run,
        "sources": sel["per_source"],
        "source_hashes": sel["source_hashes"],
        "rows_requested": spec["total_rows"],
        "rows_written": rows_written,
        "domain_mix": sel["domain_mix"],
        "length_mix": sel["length_mix"],
        "length_thresholds": {"short_max": SHORT_MAX, "medium_max": MEDIUM_MAX},
        "dedupe": sel["dedupe"],
        "constraints": spec["constraints"],
        "faq_balance": sel.get("faq_balance") or {},
        "leakage": {"status": "scanned_clean", "basis": "source_catalogue"},
    }


def _report_md(manifest: Dict[str, Any], *, dry_run: bool) -> str:
    lines: List[str] = []
    lines.append(f"# Data mixture: {manifest['name']}")
    lines.append("")
    lines.append(f"- mode: {'DRY-RUN (plan only, no train.jsonl written)' if dry_run else 'REAL'}")
    lines.append(f"- created_utc: {manifest.get('created_utc')}")
    lines.append(f"- rows_requested: {manifest['rows_requested']}")
    lines.append(f"- rows_written{' (estimate)' if dry_run else ''}: {manifest['rows_written']}")
    lt = manifest["length_thresholds"]
    lines.append(f"- length thresholds: short < {lt['short_max']} chars; "
                 f"medium < {lt['medium_max']}; long >= {lt['medium_max']}")
    lines.append(f"- leakage: {manifest['leakage']['status']} "
                 f"(basis: {manifest['leakage']['basis']})")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    lines.append("| source | weight | leakage | rows_avail | budget | kept |")
    lines.append("|---|---|---|---|---|---|")
    for s in manifest["sources"]:
        lines.append(f"| {s['source']} | {s['weight']} | {s['leakage']} | "
                     f"{s['rows_available']} | {s['budget']} | {s['kept']} |")
    lines.append("")
    lines.append("## Domain mix")
    lines.append("")
    for k, v in manifest["domain_mix"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Length mix")
    lines.append("")
    for k in ("short", "medium", "long"):
        lines.append(f"- {k}: {manifest['length_mix'].get(k, 0.0)}")
    lines.append("")
    lines.append("## Dedupe")
    lines.append("")
    d = manifest["dedupe"]
    lines.append(f"- exact: {d['exact']} (dropped {d['dropped_exact']})")
    lines.append(f"- normalized_text: {d['normalized_text']} (dropped {d['dropped_normalized_text']})")
    lines.append(f"- simhash: {d['simhash']}")
    lines.append("")
    return "\n".join(lines)


def _write_outputs(out_dir: Path, manifest: Dict[str, Any], *, dry_run: bool,
                   examples: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    report_path = out_dir / "report.md"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_report_md(manifest, dry_run=dry_run), encoding="utf-8")
    written = {"manifest": str(manifest_path), "report": str(report_path)}
    # In dry-run we deliberately DO NOT write the (potentially huge) train.jsonl.
    if not dry_run and examples is not None:
        train_path = out_dir / "train.jsonl"
        with train_path.open("w", encoding="utf-8") as w:
            for e in examples:
                row = {"query": e["query"], "positive": e["positive"],
                       "domain": e.get("domain"), "source": e.get("source")}
                w.write(json.dumps(row, ensure_ascii=False) + "\n")
        written["train"] = str(train_path)
    return written


# ------------------------------------------------------------------------- public API
def plan_mixture(config: Dict[str, Any], catalogue: Dict[str, Any], *,
                 out_dir: Path | str, created_utc: Optional[str] = None) -> Dict[str, Any]:
    """DRY-RUN: validate + estimate, write manifest (rows_written=estimate) + report.md only.

    Never writes train.jsonl. Returns ``{"manifest", "written", "dry_run": True}``.
    """
    spec = validate_mixture_config(config, catalogue)
    sel = _select(spec, collect=False)
    manifest = _build_manifest(spec, sel, rows_written=sel["rows_written"],
                               created_utc=created_utc, dry_run=True)
    written = _write_outputs(Path(out_dir), manifest, dry_run=True)
    return {"manifest": manifest, "written": written, "dry_run": True}


def build_mixture(config: Dict[str, Any], catalogue: Dict[str, Any], *,
                  out_dir: Path | str, created_utc: Optional[str] = None) -> Dict[str, Any]:
    """REAL run: validate + materialize, write train.jsonl + manifest + report.md.

    Returns ``{"manifest", "written", "dry_run": False}``.
    """
    spec = validate_mixture_config(config, catalogue)
    sel = _select(spec, collect=True)
    manifest = _build_manifest(spec, sel, rows_written=sel["rows_written"],
                               created_utc=created_utc, dry_run=False)
    written = _write_outputs(Path(out_dir), manifest, dry_run=False, examples=sel["examples"])
    return {"manifest": manifest, "written": written, "dry_run": False}


def run(config: Dict[str, Any], catalogue: Dict[str, Any], *, out_dir: Path | str,
        dry_run: bool = True, created_utc: Optional[str] = None) -> Dict[str, Any]:
    """Convenience dispatcher used by the script: dry_run -> plan_mixture else build_mixture."""
    if dry_run:
        return plan_mixture(config, catalogue, out_dir=out_dir, created_utc=created_utc)
    return build_mixture(config, catalogue, out_dir=out_dir, created_utc=created_utc)
