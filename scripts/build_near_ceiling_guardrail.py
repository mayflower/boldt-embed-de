#!/usr/bin/env python3
"""Build a held-out NEAR-CEILING guardrail set (stdlib, no ML) for validating the bounded rerank
policy WITHOUT tuning on GermanQuAD/DT-test. Selects near-ceiling queries (first-stage already
strong) from non-public, leakage-safe candidate lists. Public-eval sources are excluded; overlap
with training queries is a HARD failure. `--dry-run` writes the report only (no torch).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import near_ceiling_guardrail as NC  # noqa: E402


def _read(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]


def _train_qids(path):
    if not path:
        return set()
    ids = set()
    for r in _read(path):
        if isinstance(r, dict) and r.get("query_id") is not None:
            ids.add(str(r["query_id"]))
        elif isinstance(r, str):
            ids.add(r.strip())
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-lists", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--target-size", type=int, default=1000)
    ap.add_argument("--exclude-sources", default="germanquad,dt_test")
    ap.add_argument("--train-queries", default=None,
                    help="JSONL of {query_id} (or query_id per line) the guardrail must be disjoint from")
    ap.add_argument("--min-first-stage-ndcg", type=float, default=NC.MIN_FIRST_STAGE_NDCG)
    ap.add_argument("--min-oracle-ndcg", type=float, default=NC.MIN_ORACLE_NDCG)
    ap.add_argument("--min-candidates", type=int, default=NC.MIN_CANDIDATES)
    ap.add_argument("--min-sources", type=int, default=NC.MIN_SOURCES)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lp = pathlib.Path(args.candidate_lists)
    if not lp.exists():
        print(f"ERROR: candidate lists not found: {lp}", file=sys.stderr)
        return 2
    rows = _read(lp)
    res = NC.build(rows, exclude_sources=set(args.exclude_sources.split(",")),
                   train_query_ids=_train_qids(args.train_queries), target_size=args.target_size,
                   min_fs=args.min_first_stage_ndcg, min_oracle=args.min_oracle_ndcg,
                   min_candidates=args.min_candidates, min_sources=args.min_sources)
    report = res["report"]

    if args.dry_run:
        assert "torch" not in sys.modules, "dry-run must not import torch"
    pathlib.Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    print(f"[near-ceiling] status={report['status']} selected={report['num_selected']} "
          f"(target {report['target_size']}); excluded_public={report['excluded_public_or_listed_source']} "
          f"train_overlap={report['training_overlap']['overlap_count']} "
          f"multi_source={report['multi_source_fraction']} -> {args.report}")
    for e in report["errors"]:
        print(f"  ✗ {e}", file=sys.stderr)
    if report["status"] != "pass":
        print("FAIL — near-ceiling guardrail not built (see report)", file=sys.stderr)
        return 1

    if args.dry_run:
        print("dry-run-ok (no ML; report written, guardrail set NOT written)")
        return 0
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in res["selected"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[near-ceiling] wrote {len(res['selected'])} near-ceiling guardrail lists -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
