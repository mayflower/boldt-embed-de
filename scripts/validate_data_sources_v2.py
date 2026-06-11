#!/usr/bin/env python3
"""Validate the v2 data-source manifest and print a Markdown table (pure stdlib, no network).

Exits non-zero if any source is invalid or violates the fail-closed training-eligibility rules
(missing license, public_benchmark/eval_only marked training-allowed, uncertain license, etc.).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import source_manifest as sm  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(ROOT / "configs" / "data_sources_v2.json"))
    ap.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = ap.parse_args()

    if not pathlib.Path(args.manifest).exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    d = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    errors = sm.validate_source_manifest(d)
    if errors:
        print(f"INVALID manifest ({len(errors)} problem(s)):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    entries = sm.load_source_manifest(args.manifest)
    if args.format == "json":
        print(json.dumps({"status": "ok", "n_sources": len(entries),
                          "n_training": len(sm.training_sources(entries)),
                          "training_source_ids": [e.source_id for e in sm.training_sources(entries)]},
                         ensure_ascii=False, indent=2))
    else:
        print(sm.render_markdown(entries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
