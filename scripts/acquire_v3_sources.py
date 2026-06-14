#!/usr/bin/env python3
"""Acquire v3 real-domain sources (admin/FAQ/legal + web/wiki/stress/cross-lingual).

Fail-closed: a source trains only if its license is verified and it is not a benchmark /
eval-only / overlap-risky. ``dry-run`` and ``materialize-local`` never touch the network;
``download-hf`` is the only mode that may pull hf_dataset sources (lazy import).

Modes:
  dry-run            validate the manifest + plan; write a planned summary; no data I/O of corpora
  materialize-local  read+validate local_jsonl / local_corpus_jsonl drops, write to --output-dir
  download-hf        additionally pull hf_dataset sources (not run in CI/tests)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import domain_source_acquisition as dsa  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(ROOT / "configs" / "data_sources_v3.json"))
    ap.add_argument("--output-dir", default=str(ROOT / "data" / "raw" / "v3"))
    ap.add_argument("--mode", choices=["dry-run", "materialize-local", "download-hf"],
                    default="dry-run")
    ap.add_argument("--fail-on-unverified-license", action="store_true",
                    help="exit non-zero if ANY source has license_verified=false (strict gate)")
    ap.add_argument("--summary-output", default=None, help="write the acquisition summary JSON here")
    args = ap.parse_args()

    try:
        entries = dsa.load_v3_manifest(args.manifest)   # fail-closed: raises on invalid manifest
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError:
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    summary = dsa.acquire(entries, args.output_dir, args.mode,
                          fail_on_unverified_license=args.fail_on_unverified_license)
    # dry-run must not have imported any ML/network library
    if args.mode == "dry-run":
        assert "torch" not in sys.modules and "datasets" not in sys.modules, \
            "dry-run must not import ML/network libraries"

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary_output:
        pathlib.Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")

    print(f"[v3-acquire] mode={args.mode} status={summary['status']} "
          f"materialized={len(summary['materialized'])} blocked={len(summary['blocked'])} "
          f"real_domains_missing={summary['real_domains_missing']}", file=sys.stderr)
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
