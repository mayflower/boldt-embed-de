#!/usr/bin/env python3
"""ar_build_mixture.py — build a constraint-aware training-data mixture (CLI).

Thin CLI over ``boldt_embed.data_mixture_optimizer``. PURE STDLIB. ``--dry-run`` is the
default behaviour intent: it validates the config against the catalogue and writes ONLY a
plan/report (``manifest.json`` with ``rows_written`` = estimate, and ``report.md``) — it
never writes the large ``train.jsonl``. Drop ``--dry-run`` (i.e. pass ``--no-dry-run``) to
actually materialize the corpus.

Example (dry-run against the real catalogue):

    conda run -n boldtembed python scripts/ar_build_mixture.py \
        --config configs/autoresearch/mixtures/v8_balanced.json \
        --catalog configs/data_sources.json \
        --out outputs/autoresearch/mixtures/v8_balanced \
        --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make ``src/`` importable when run as a script (no install required).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from boldt_embed.data_mixture_optimizer import (  # noqa: E402
    MixtureConfigError,
    load_catalogue,
    run as run_mixture,
)


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a constraint-aware training-data mixture.")
    ap.add_argument("--config", required=True, help="mixture config JSON (name/total_rows/sources/constraints)")
    ap.add_argument("--catalog", default="configs/data_sources.json",
                    help="data-sources catalogue JSON (default: configs/data_sources.json)")
    ap.add_argument("--out", required=True, help="output directory for manifest/report(/train.jsonl)")
    ap.add_argument("--created-utc", default=None,
                    help="optional fixed UTC timestamp recorded in the manifest (kept injectable "
                         "for deterministic provenance / tests)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--dry-run", dest="dry_run", action="store_true", default=True,
                     help="plan only: write manifest(estimate)+report, NOT train.jsonl (default)")
    grp.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                     help="REAL run: materialize and write train.jsonl")
    ap.add_argument("--format", choices=("markdown", "json"), default="markdown",
                    help="stdout summary format (default: markdown)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ERROR: could not read mixture config {args.config!r}: {exc}", file=sys.stderr)
        return 2
    try:
        catalogue = load_catalogue(args.catalog)
    except MixtureConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        result = run_mixture(config, catalogue, out_dir=args.out,
                             dry_run=args.dry_run, created_utc=args.created_utc)
    except MixtureConfigError as exc:
        # Fail-closed: name the offending source / problem and exit non-zero.
        print(f"ERROR (fail-closed): {exc}", file=sys.stderr)
        return 1

    manifest = result["manifest"]
    written = result["written"]
    if args.format == "json":
        print(json.dumps({"dry_run": result["dry_run"], "written": written,
                          "manifest": manifest}, ensure_ascii=False, indent=2))
    else:
        mode = "DRY-RUN (no train.jsonl written)" if result["dry_run"] else "REAL"
        print(f"# Mixture '{manifest['name']}' — {mode}")
        print(f"rows_requested={manifest['rows_requested']} "
              f"rows_written{'(estimate)' if result['dry_run'] else ''}={manifest['rows_written']}")
        print(f"domain_mix={json.dumps(manifest['domain_mix'], ensure_ascii=False)}")
        print(f"length_mix={json.dumps(manifest['length_mix'], ensure_ascii=False)}")
        print(f"dedupe={json.dumps(manifest['dedupe'], ensure_ascii=False)}")
        for k, v in written.items():
            print(f"wrote {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
