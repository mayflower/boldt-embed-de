#!/usr/bin/env python3
"""Dense gate for dense-v6.1: decides whether the Boldt dense RAG embedder can be **recommended for
German RAG first-stage retrieval**. Based on DENSE RETRIEVAL QUALITY only — never policy or reranker
behavior. The dense recommendation is INDEPENDENT of the reranker (which stays experimental unless
its own raw gate passes). Pure stdlib.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

GATE = {
    "webfaq_recall_at_50_min": 0.90,
    "webfaq_recall_at_100_min": 0.96,
    "webfaq_missing_positive_rate_max": 0.04,
    "webfaq_ndcg_at_10_min": 0.67,
    "germanquad_ndcg_at_10_min": 0.88,
    "dt_test_ndcg_at_10_min": 0.94,
    "matryoshka_256_retention_min": 0.95,
}


def dense_gate(metrics: dict, *, public_eval_leakage: bool = False) -> dict:
    """metrics: {webfaq:{recall@50,recall@100,missing_positive_rate,ndcg@10,matryoshka_256_retention},
    germanquad:{ndcg@10}, dt_test:{ndcg@10}}. Returns the gate decision (dense-only)."""
    checks = []

    def chk(name, ok, detail):
        checks.append({"check": name, "status": "pass" if ok else "fail", "detail": detail})

    wf = metrics.get("webfaq") or {}
    chk("webfaq_recall_at_50", wf.get("recall@50") is not None
        and wf["recall@50"] >= GATE["webfaq_recall_at_50_min"] - 1e-9,
        f"{wf.get('recall@50')} (min {GATE['webfaq_recall_at_50_min']})")
    chk("webfaq_recall_at_100", wf.get("recall@100") is not None
        and wf["recall@100"] >= GATE["webfaq_recall_at_100_min"] - 1e-9,
        f"{wf.get('recall@100')} (min {GATE['webfaq_recall_at_100_min']})")
    chk("webfaq_missing_positive_rate", wf.get("missing_positive_rate") is not None
        and wf["missing_positive_rate"] <= GATE["webfaq_missing_positive_rate_max"] + 1e-9,
        f"{wf.get('missing_positive_rate')} (max {GATE['webfaq_missing_positive_rate_max']})")
    chk("webfaq_ndcg_at_10", wf.get("ndcg@10") is not None
        and wf["ndcg@10"] >= GATE["webfaq_ndcg_at_10_min"] - 1e-9,
        f"{wf.get('ndcg@10')} (min {GATE['webfaq_ndcg_at_10_min']})")
    gq = metrics.get("germanquad") or {}
    chk("germanquad_ndcg_at_10", gq.get("ndcg@10") is not None
        and gq["ndcg@10"] >= GATE["germanquad_ndcg_at_10_min"] - 1e-9,
        f"{gq.get('ndcg@10')} (min {GATE['germanquad_ndcg_at_10_min']})")
    dt = metrics.get("dt_test") or {}
    chk("dt_test_ndcg_at_10", dt.get("ndcg@10") is not None
        and dt["ndcg@10"] >= GATE["dt_test_ndcg_at_10_min"] - 1e-9,
        f"{dt.get('ndcg@10')} (min {GATE['dt_test_ndcg_at_10_min']})")
    ret = wf.get("matryoshka_256_retention")
    chk("matryoshka_256_retention", ret is not None and ret >= GATE["matryoshka_256_retention_min"] - 1e-9,
        f"{ret} (min {GATE['matryoshka_256_retention_min']})")
    chk("no_public_eval_leakage", not public_eval_leakage, "no public-eval leakage")

    failing = [c for c in checks if c["status"] == "fail"]
    status = "pass" if not failing else "fail"
    if status == "pass":
        recommendation = ("Dense embedder CAN be recommended for German RAG first-stage retrieval. "
                          "The reranker remains experimental unless the raw reranker gate passes "
                          "(independent decision).")
    else:
        recommendation = ("Do NOT recommend the dense embedder yet. Failed dense targets: "
                          + ", ".join(f"{c['check']} ({c['detail']})" for c in failing))
    return {"status": status, "checks": checks, "failing": failing, "thresholds": GATE,
            "recommendation": recommendation, "failed_targets": [c["check"] for c in failing],
            "independent_of_reranker": True, "based_on": "dense retrieval quality only"}


def _extract(summary: dict, model: str) -> dict:
    """Pull the gate metrics for ``model`` from an eval summary {model:{set:metrics}}."""
    m = summary.get(model) or {}
    out = {}
    for s in ("webfaq", "germanquad", "dt_test"):
        if s in m:
            r = m[s]
            out[s] = {"recall@50": r.get("recall@50"), "recall@100": r.get("recall@100"),
                      "missing_positive_rate": r.get("missing_positive_rate"),
                      "ndcg@10": r.get("ndcg@10"),
                      "matryoshka_256_retention": r.get("matryoshka_256_retention")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="outputs/v6-1-dense-top50/dense_eval_summary.json")
    ap.add_argument("--model", default="dense-v6.1")
    ap.add_argument("--output", default="outputs/v6-1-dense-top50/dense_gate.json")
    ap.add_argument("--markdown", default="outputs/v6-1-dense-top50/dense_gate.md")
    ap.add_argument("--public-eval-leakage", action="store_true")
    args = ap.parse_args()

    summary = json.loads(pathlib.Path(args.summary).read_text(encoding="utf-8"))
    metrics = _extract(summary, args.model)
    gate = dense_gate(metrics, public_eval_leakage=args.public_eval_leakage)
    gate["model"] = args.model
    gate["metrics"] = metrics

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(gate, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    md = [f"# Dense gate ({args.model}): **{gate['status']}**", "",
          "_Decides whether the Boldt dense RAG embedder can be recommended for German RAG "
          "first-stage retrieval — dense retrieval quality only, INDEPENDENT of the reranker._", "",
          "| check | status | detail |", "|---|---|---|"]
    for c in gate["checks"]:
        md.append(f"| {c['check']} | {'✅' if c['status'] == 'pass' else '❌'} | {c['detail']} |")
    md += ["", f"**{gate['recommendation']}**"]
    pathlib.Path(args.markdown).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[v6.1-dense-gate] {args.model}: status={gate['status']} "
          f"failed={gate['failed_targets']} -> {args.output}")
    return 0 if gate["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
