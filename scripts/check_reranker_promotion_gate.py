#!/usr/bin/env python3
"""Reranker promotion gate (pure stdlib): a reranker may NOT be promoted if it degrades any
held-out set. Reads lift reports (first-stage vs +reranker nDCG@10), computes deltas, and
FAILS if DT-test delta < 0 or GermanQuAD delta < 0 (the v1 failure mode). +0.02 on GermanQuAD
is the target (reported separately). Exit 0 = pass, 1 = fail.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


def _delta(report_path: str):
    d = json.loads(pathlib.Path(report_path).read_text(encoding="utf-8"))
    fs = next((v for k, v in d.items() if k.startswith("first_stage_ndcg@")), None)
    rr = next((v for k, v in d.items() if k.startswith("student_reranker_ndcg@")), None)
    if fs is None or rr is None:
        raise ValueError(f"{report_path}: missing first_stage/student_reranker ndcg keys")
    return round(rr - fs, 4), fs, rr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dt-test", required=True, help="reranker lift report JSON for DT-test")
    ap.add_argument("--germanquad", required=True, help="reranker lift report JSON for GermanQuAD")
    ap.add_argument("--additional", nargs="*", default=None, help="extra lift reports (delta>=0 too)")
    ap.add_argument("--germanquad-target", type=float, default=0.02)
    ap.add_argument("--catastrophic-degradation", type=float, default=0.02,
                    help="any evaluated domain dropping by more than this is a catastrophic fail")
    ap.add_argument("--training-summary", default=None,
                    help="reranker training summary JSON; gate fails if positives are low-precision")
    ap.add_argument("--allow-low-precision-positives", action="store_true",
                    help="override the high-precision-positives requirement")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cat = args.catastrophic_degradation
    dt_delta, dt_fs, dt_rr = _delta(args.dt_test)
    gq_delta, gq_fs, gq_rr = _delta(args.germanquad)
    checks = {
        "dt_test_delta_nonneg": dt_delta >= 0.0,
        "germanquad_delta_nonneg": gq_delta >= 0.0,           # hard floor — the v1/v2 lesson
        "dt_test_not_catastrophic": dt_delta >= -cat,
        "germanquad_not_catastrophic": gq_delta >= -cat,
    }
    extra = []
    for p in args.additional or []:
        dd, _, _ = _delta(p)
        stem = pathlib.Path(p).stem
        extra.append({"report": p, "delta": dd, "nonneg": dd >= 0.0, "not_catastrophic": dd >= -cat})
        checks[f"extra_{stem}_nonneg"] = dd >= 0.0
        checks[f"extra_{stem}_not_catastrophic"] = dd >= -cat
    # high-precision-positives requirement (from the training summary), overridable.
    high_precision = None
    if args.training_summary and pathlib.Path(args.training_summary).exists():
        ts = json.loads(pathlib.Path(args.training_summary).read_text(encoding="utf-8"))
        high_precision = bool(ts.get("high_precision_positives", False)) and \
            float(ts.get("positive_threshold", 0)) >= 4.0
        checks["high_precision_positives"] = high_precision or args.allow_low_precision_positives
    passed = all(checks.values())
    result = {
        "status": "pass" if passed else "fail",
        "dt_test": {"first_stage": dt_fs, "reranked": dt_rr, "delta": dt_delta},
        "germanquad": {"first_stage": gq_fs, "reranked": gq_rr, "delta": gq_delta,
                       "target": args.germanquad_target,
                       "target_met": gq_delta >= args.germanquad_target},
        "additional": extra,
        "catastrophic_degradation_threshold": cat,
        "high_precision_positives": high_precision,
        "checks": checks,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    verdict = "PASS — reranker may be promoted" if passed else \
        "FAIL — reranker degrades a held-out set; do NOT promote"
    print(f"\n{verdict}: DT-test Δ={dt_delta:+.4f}, GermanQuAD Δ={gq_delta:+.4f}")
    if args.output:
        pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
