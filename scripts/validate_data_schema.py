#!/usr/bin/env python3
"""Validate a JSONL training-data file against the training-pair schema (pure stdlib).

Checks: record/dataset validity, allowed licenses, PII, and (optionally) benchmark leakage
against an eval-corpus file. Implements the prompt-04 / ADR-004 data gate.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data as datamod  # noqa: E402

DEFAULTS = [
    ROOT / "data" / "samples" / "toy_triples_de.jsonl",
    ROOT / "data" / "samples" / "toy_pairs_de.jsonl",
]


def validate_file(path: pathlib.Path, eval_corpus=None) -> dict:
    records = datamod.load_jsonl(path)
    report = datamod.validate_dataset(records)
    disallowed = datamod.check_licenses(records)
    pii = datamod.scan_pii(records)
    leakage = datamod.find_leakage(records, eval_corpus) if eval_corpus else []
    ok = report.ok and not disallowed and not pii and not leakage
    return {
        "file": str(path.relative_to(ROOT)),
        "status": "pass" if ok else "fail",
        "num_records": report.num_records,
        "errors": report.errors,
        "disallowed_licenses": disallowed,
        "pii_hits": pii,
        "leakage_hits": leakage,
        "licenses": report.licenses,
        "neg_type_counts": report.neg_type_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="append", help="JSONL file(s); default = toy samples")
    parser.add_argument("--eval-corpus", help="JSONL/txt of eval texts for leakage check")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    args = parser.parse_args()

    files = [pathlib.Path(p) for p in (args.data or DEFAULTS)]
    eval_texts = None
    if args.eval_corpus:
        eval_texts = [l for l in pathlib.Path(args.eval_corpus).read_text(
            encoding="utf-8").splitlines() if l.strip()]

    reports = [validate_file(f, eval_texts) for f in files]
    overall = "pass" if all(r["status"] == "pass" for r in reports) else "fail"
    out = {"status": overall, "files": reports}

    if args.format == "markdown":
        lines = ["# Data Schema Validation", "", f"Status: **{overall}**", ""]
        for r in reports:
            lines.append(f"## {r['file']} — {r['status']}")
            lines.append(f"- records: {r['num_records']}, licenses: {r['licenses']}")
            for key in ("errors", "disallowed_licenses", "pii_hits", "leakage_hits"):
                if r[key]:
                    lines.append(f"- {key}: {r[key]}")
            lines.append("")
        print("\n".join(lines))
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
