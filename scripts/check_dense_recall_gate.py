#!/usr/bin/env python3
"""DENSE-RECALL STOP GATE: a reranker cannot rank a document the first stage never retrieved. If
first-stage recall is insufficient, this gate FAILS and writes STOP_RERANKER_TRAINING.md so no GPU
is wasted training/evaluating another reranker on lists that lack the positives. Pure stdlib.

Inputs (any subset; metrics are extracted defensively):
  --recall-report   dense-vs-bm25 recall json  (e.g. outputs/v6-dense-rag/webfaq_real_recall_bm25_vs_dense.json)
  --union-report    candidate-union report      (e.g. outputs/v6-reranker/eval-lists/webfaq_union_report.json)
  --audit-report    first-stage audit json       (e.g. outputs/v6-dense-rag/first_stage_audit_webfaq.json)
  --local-rag-report  optional local-RAG recall json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

DEFAULT_TARGETS = {
    "recall_at_100": 0.95,
    "positive_in_top_50": 0.90,
    "oracle_ndcg_at_10": 0.95,
    "max_missing_positive_rate": 0.10,
    "min_candidate_union_size": 20,
    "local_rag_recall_at_100": 0.90,
}
STOP_FILE = "STOP_RERANKER_TRAINING.md"
STOP_MESSAGE = ("Reranker cannot recover missing positives. "
                "Improve dense retrieval or candidate generation first.")


def dense_recall_gate(metrics: dict, *, targets: dict = None) -> dict:
    """metrics: {set_name: {recall_at_100, positive_in_top_50, oracle_ndcg_at_10,
    candidate_union_size, missing_positive_rate}}. WebFAQ is required. Pure function."""
    t = {**DEFAULT_TARGETS, **(targets or {})}
    checks = []

    def chk(name, ok, detail):
        checks.append({"check": name, "status": "pass" if ok else "fail", "detail": detail})

    wf = metrics.get("webfaq")
    if not wf:
        chk("webfaq_metrics_present", False, "no WebFAQ recall metrics provided — cannot decide")
    else:
        def g(k):
            return wf.get(k)
        chk("webfaq_recall_at_100", g("recall_at_100") is not None
            and g("recall_at_100") >= t["recall_at_100"] - 1e-9,
            f"{g('recall_at_100')} (min {t['recall_at_100']})")
        chk("webfaq_positive_in_top_50", g("positive_in_top_50") is not None
            and g("positive_in_top_50") >= t["positive_in_top_50"] - 1e-9,
            f"{g('positive_in_top_50')} (min {t['positive_in_top_50']})")
        chk("webfaq_oracle_ndcg_at_10", g("oracle_ndcg_at_10") is not None
            and g("oracle_ndcg_at_10") >= t["oracle_ndcg_at_10"] - 1e-9,
            f"{g('oracle_ndcg_at_10')} (min {t['oracle_ndcg_at_10']})")
        if g("candidate_union_size") is not None:
            chk("candidate_union_size", g("candidate_union_size") >= t["min_candidate_union_size"],
                f"{g('candidate_union_size')} (min {t['min_candidate_union_size']})")
        if g("missing_positive_rate") is not None:
            chk("missing_positive_rate", g("missing_positive_rate")
                <= t["max_missing_positive_rate"] + 1e-9,
                f"{g('missing_positive_rate')} (max {t['max_missing_positive_rate']})")
    lr = metrics.get("local_rag")
    if lr and lr.get("recall_at_100") is not None:
        chk("local_rag_recall_at_100", lr["recall_at_100"] >= t["local_rag_recall_at_100"] - 1e-9,
            f"{lr['recall_at_100']} (min {t['local_rag_recall_at_100']})")

    failing = [c for c in checks if c["status"] == "fail"]
    # "positives absent" = the recall-sufficiency checks failed (not merely a ranking-quality miss)
    absent_checks = {"webfaq_recall_at_100", "missing_positive_rate", "webfaq_oracle_ndcg_at_10",
                     "webfaq_metrics_present", "local_rag_recall_at_100"}
    positives_absent = any(c["check"] in absent_checks for c in failing)
    return {"status": "pass" if not failing else "fail", "checks": checks, "failing": failing,
            "targets": t, "positives_absent": positives_absent,
            "stop_reason": STOP_MESSAGE if failing else None}


# ----------------------------------------------------------------- report extraction
def _read(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8")) if p and pathlib.Path(p).exists() else None


def extract_webfaq_metrics(recall_report=None, union_report=None, audit_report=None) -> dict:
    """Pull the gate metrics from whatever reports exist. Prefers the dense retriever's recall over
    the real corpus; falls back to the candidate-union / audit reports."""
    m = {}
    rr = _read(recall_report)
    if rr:
        dense = rr.get("dense_v6") or rr.get("dense") or {}
        if dense.get("recall@100") is not None:
            m["recall_at_100"] = dense["recall@100"]
        if dense.get("recall@50") is not None:
            m["positive_in_top_50"] = dense["recall@50"]
    ur = _read(union_report)
    if ur:
        uni = ur.get("union_recall") or {}
        m.setdefault("recall_at_100", uni.get("recall@200", uni.get("recall@100")))
        m.setdefault("positive_in_top_50", uni.get("recall@50"))
        if ur.get("positive_present_rate") is not None:
            # oracle nDCG@10 ceiling ~= present_rate (a present positive can be placed at rank 0)
            m.setdefault("oracle_ndcg_at_10", ur["positive_present_rate"])
            m.setdefault("missing_positive_rate", round(1.0 - ur["positive_present_rate"], 6))
        if ur.get("list_size") is not None:
            m.setdefault("candidate_union_size", ur["list_size"])
    ar = _read(audit_report)
    if ar:
        m.setdefault("oracle_ndcg_at_10", ar.get("oracle_ndcg10_retriever"))
        if ar.get("missing_positive_rate") is not None:
            m.setdefault("missing_positive_rate", ar["missing_positive_rate"])
        rec = ar.get("recall") or {}
        m.setdefault("recall_at_100", rec.get("recall@100"))
    return {k: v for k, v in m.items() if v is not None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recall-report", default=None)
    ap.add_argument("--union-report", default=None)
    ap.add_argument("--audit-report", default=None)
    ap.add_argument("--local-rag-report", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--stop-file", default=str(pathlib.Path(__file__).resolve().parents[1] / STOP_FILE))
    ap.add_argument("--target-recall-100", type=float, default=DEFAULT_TARGETS["recall_at_100"])
    ap.add_argument("--target-top-50", type=float, default=DEFAULT_TARGETS["positive_in_top_50"])
    ap.add_argument("--target-oracle", type=float, default=DEFAULT_TARGETS["oracle_ndcg_at_10"])
    args = ap.parse_args()

    metrics = {"webfaq": extract_webfaq_metrics(args.recall_report, args.union_report,
                                                args.audit_report)}
    lr = _read(args.local_rag_report)
    if lr:
        dense = lr.get("dense_v6") or lr.get("dense") or lr
        if dense.get("recall@100") is not None:
            metrics["local_rag"] = {"recall_at_100": dense["recall@100"]}
    targets = {"recall_at_100": args.target_recall_100, "positive_in_top_50": args.target_top_50,
               "oracle_ndcg_at_10": args.target_oracle}
    gate = dense_recall_gate(metrics, targets=targets)
    gate["metrics"] = metrics

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(gate, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    stop = pathlib.Path(args.stop_file)
    # The BLOCKING stop file is written only when positives are genuinely ABSENT (the acceptance
    # condition: don't waste GPU when the reranker cannot recover positives). A miss on a ranking-
    # quality target alone (e.g. top-50) with positives present is ADVISORY, never blocking.
    if gate["positives_absent"]:
        lines = [f"# STOP — reranker training blocked ({STOP_FILE})", "",
                 STOP_MESSAGE, "",
                 f"The dense-recall gate FAILED (`{args.output}`). First-stage recall is insufficient "
                 "for the reranker to recover positives. Do NOT train or evaluate more rerankers until "
                 "this passes (or pass `--force-research-run`, which marks the run "
                 "`invalid_for_promotion`).", "",
                 f"**Positives genuinely absent:** {gate['positives_absent']}", "",
                 "## Failing checks", ""]
        for c in gate["failing"]:
            lines.append(f"- ❌ {c['check']}: {c['detail']}")
        lines += ["", "## WebFAQ metrics", ""]
        for k, v in sorted(metrics.get("webfaq", {}).items()):
            lines.append(f"- {k}: {v}")
        lines += ["", "## Fix first", "",
                  "- Improve the dense retriever (recall@100 / top-50) or candidate generation.",
                  "- Re-run `scripts/audit_first_stage_recall.py` + `scripts/build_v6_candidate_union.py`.",
                  "- Re-run this gate; it removes this file when recall passes."]
        stop.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[dense-recall-gate] STOP (positives absent) failing="
              f"{[c['check'] for c in gate['failing']]} -> wrote {stop}")
        return 1
    # positives present: never leave a blocking STOP file
    if stop.exists():
        stop.unlink()
    if gate["status"] == "fail":
        print(f"[dense-recall-gate] ADVISORY (NOT blocking — positives present, recall sufficient): "
              f"targets missed {[c['check'] for c in gate['failing']]}. No STOP file written; "
              f"reranker training is allowed but the listed targets are below threshold.")
        return 1
    print(f"[dense-recall-gate] PASS metrics={metrics.get('webfaq')} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
