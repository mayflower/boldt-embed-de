#!/usr/bin/env python3
"""Summarize the v4 German RAG reranker run into V4_RAG_RESULTS.{md,json} (stdlib, no ML).

Reads a v4 work-dir (reranker_lift_*.json, rag_reranker_gate.json, teacher summary, candidate
lists) and produces an honest executive verdict + the 7 report sections. Promotion reflects RAG
usefulness; GerDaLIR is diagnostic-only.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import rag_reranker_eval as RE  # noqa: E402

PUB = ("germanquad", "dt_test")


def _load(path):
    p = pathlib.Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:
        return None


def _count_lines(path):
    p = pathlib.Path(path)
    return sum(1 for l in p.read_text(encoding="utf-8").split("\n") if l.strip()) if p.exists() else 0


def summarize(work: pathlib.Path) -> dict:
    eval_dir = work / "eval"
    reports = []
    for p in sorted(eval_dir.glob("reranker_lift_*.json")):
        r = _load(p)
        if isinstance(r, dict) and "eval_set" in r:
            reports.append(r)
    by_set = {r["eval_set"]: r for r in reports}

    gate = _load(eval_dir / "rag_reranker_gate.json")
    if gate is None and reports:
        gate = RE.evaluate_promotion(reports)
    gate_pass = bool(gate and gate.get("status") == "pass")

    # verdict: promoted / mixed / not_promoted
    webfaq_delta = by_set.get("webfaq", {}).get("delta_ndcg@10")
    if gate_pass:
        verdict = "promoted"
    elif webfaq_delta is not None and webfaq_delta >= RE.WEBFAQ_MIN_DELTA:
        verdict = "mixed"        # RAG lift is there but a neutral/fixed/recall check failed
    else:
        verdict = "not_promoted"

    teacher_summary = _load(work / "teacher" / "rag_teacher_summary.json") or {}
    train_lists = work / "candidate_lists" / "rag_reranker_train_lists.jsonl"
    train_scored = work / "teacher" / "rag_train_scored.jsonl"

    def _row(r):
        return {"eval_set": r["eval_set"], "diagnostic": r.get("diagnostic", False),
                "first_stage_ndcg@10": r.get("first_stage_ndcg@10"),
                "reranked_ndcg@10": r.get("reranked_ndcg@10"),
                "delta_ndcg@10": r.get("delta_ndcg@10"),
                "first_stage_mrr@10": r.get("first_stage_mrr@10"),
                "reranked_mrr@10": r.get("reranked_mrr@10"),
                "positive_in_top_10_before": r.get("positive_in_top_10_before"),
                "positive_in_top_10_after": r.get("positive_in_top_10_after"),
                "answer_support_at_10": r.get("answer_support_at_10"),
                "oracle_ndcg@10": r.get("oracle_ndcg@10"),
                "first_stage_recall_top_10": r.get("first_stage_recall_top_10"),
                "fixed_candidates": r.get("fixed_candidates")}

    failures = [r["eval_set"] for r in reports
                if not r.get("diagnostic") and (r.get("delta_ndcg@10") or 0) < 0]
    calibration = [r["eval_set"] for r in reports
                   if (r.get("positive_in_top_10_after") or 0) < (r.get("positive_in_top_10_before") or 0)]

    return {
        "verdict": verdict,
        "gate_status": (gate or {}).get("status", "missing"),
        "training_data": {
            "webfaq_train_lists": _count_lines(train_lists),
            "teacher_scored_lists": _count_lines(train_scored),
            "hard_negatives": teacher_summary.get("negatives"),
            "gold_positives": teacher_summary.get("positives"),
            "uncertain": teacher_summary.get("uncertain"),
            "excluded_eval_splits": ["webfaq_heldout"] + list(PUB)
            + (["local_rag"] if "local_rag" in by_set else []),
        },
        "reranker_lift": {r["eval_set"]: _row(r) for r in reports},
        "first_stage_recall": {r["eval_set"]: {
            "positive_in_top_10": r.get("first_stage_recall_top_10"),
            "oracle_ndcg@10": r.get("oracle_ndcg@10")} for r in reports},
        "teacher_student": {r["eval_set"]: {
            "boldt_reranker_v4_ndcg@10": r.get("reranked_ndcg@10"),
            "qwen_teacher_ndcg@10": r.get("teacher_reranker_ndcg@10")} for r in reports},
        "failure_cases": {"reranker_hurts": failures,
                          "score_calibration_suspect": calibration},
        "decision": ("Recommend the reranker for German FAQ/RAG reranking over fixed first-stage "
                     "candidates." if verdict == "promoted"
                     else "Keep the reranker DISABLED for production; experimental only."),
        "diagnostic_sets": [r["eval_set"] for r in reports if r.get("diagnostic")],
    }


def render_markdown(s: dict) -> str:
    td = s["training_data"]
    L = ["# v4 German RAG reranker — results", "",
         f"## 1. Executive verdict: **{s['verdict']}**  (promotion gate: {s['gate_status']})", "",
         s["decision"], "",
         "## 2. Training data", "",
         f"- WebFAQ train candidate lists: {td['webfaq_train_lists']}",
         f"- teacher-scored lists: {td['teacher_scored_lists']}",
         f"- gold positives / hard negatives / uncertain: {td['gold_positives']} / "
         f"{td['hard_negatives']} / {td['uncertain']}",
         f"- excluded eval splits (never trained): {', '.join(td['excluded_eval_splits'])}", "",
         "## 3. Reranker lift (nDCG@10 over FIXED candidates)", "",
         "| eval set | first stage | reranked | delta | diagnostic |",
         "|---|--:|--:|--:|:--:|"]
    for name, r in s["reranker_lift"].items():
        L.append(f"| {name} | {r['first_stage_ndcg@10']} | {r['reranked_ndcg@10']} | "
                 f"{r['delta_ndcg@10']:+} | {'yes' if r['diagnostic'] else ''} |")
    L += ["", "## 4. First-stage recall", "", "| eval set | positive_in_top_10 | oracle_ndcg@10 |",
          "|---|--:|--:|"]
    for name, r in s["first_stage_recall"].items():
        L.append(f"| {name} | {r['positive_in_top_10']} | {r['oracle_ndcg@10']} |")
    L += ["", "## 5. Teacher / student (nDCG@10)", "",
          "| eval set | Boldt v4 | Qwen teacher |", "|---|--:|--:|"]
    for name, r in s["teacher_student"].items():
        L.append(f"| {name} | {r['boldt_reranker_v4_ndcg@10']} | "
                 f"{r['qwen_teacher_ndcg@10'] if r['qwen_teacher_ndcg@10'] is not None else 'n/a'} |")
    L += ["", "## 6. Failure cases", "",
          f"- reranker hurts (delta < 0): {s['failure_cases']['reranker_hurts'] or 'none'}",
          f"- score-calibration suspect (positive_in_top_10 dropped): "
          f"{s['failure_cases']['score_calibration_suspect'] or 'none'}", "",
          "## 7. Decision", "", s["decision"]]
    if s["diagnostic_sets"]:
        L += ["", f"_Diagnostic-only (never gates): {', '.join(s['diagnostic_sets'])}._"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--json-output", required=True)
    args = ap.parse_args()

    s = summarize(pathlib.Path(args.work_dir))
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(render_markdown(s), encoding="utf-8")
    pathlib.Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.json_output).write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v4-summary] verdict={s['verdict']} gate={s['gate_status']} "
          f"-> {args.output}, {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
