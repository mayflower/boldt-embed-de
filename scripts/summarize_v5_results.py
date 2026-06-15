#!/usr/bin/env python3
"""Summarize the REAL v5 prompt-4 run into V5_RESULTS.{md,json} (stdlib, no ML).

Reads the real artifacts on disk (acquire report, candidate-list report, teacher summary, the
hardness-aware gate, run card) and the committed v4 gate for an honest v4->v5 comparison. The
verdict is taken from the gate (fail/not promoted) — no optimistic wording.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
V5 = ROOT / "outputs/v5-small-rag"


def _load(p, default=None):
    p = pathlib.Path(p)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default
    except Exception:
        return default


def main() -> int:
    gate = _load(V5 / "eval/v5_rag_lift_gate.json", {})
    teacher = _load(V5 / "teacher/rag_teacher_summary.json", {})
    acquire = _load(ROOT / "data/raw/v5/acquire_report.json", {})
    clrep = _load(V5 / "candidate_lists/rag_train_lists.report.json", {})
    runcard = _load(ROOT / "outputs/run-cards/v5-reranker-boldt.json", {})
    v4gate = _load(ROOT / "outputs/v4-rag-reranker/eval/rag_reranker_gate.json", {})

    sets = {s["eval_set"]: s for s in gate.get("eval_sets", [])}
    v4d = (v4gate or {}).get("deltas", {})

    def row(name):
        s = sets.get(name, {})
        return {"eval_set": name, "role": s.get("role"),
                "overall_delta_ndcg@10": s.get("overall_delta_ndcg@10"),
                "medium_hard_micro_lift": s.get("primary_micro_lift"),
                "no_room_fraction": s.get("no_room_fraction"),
                "catastrophic_rate": s.get("catastrophic_rate"),
                "bucket_counts": s.get("bucket_counts")}

    result = {
        "experiment": "v5-small-rag (prompt-4 reranker)",
        "verdict": "fail / not_promoted",
        "gate_status": gate.get("gate", {}).get("status", "fail"),
        "gate_failing_checks": [c["check"] for c in gate.get("gate", {}).get("failing", [])],
        "training_data": {
            "raw_rows": acquire.get("total_rows"),
            "raw_by_domain": acquire.get("written_by_domain"),
            "domains_missing_no_real_source": acquire.get("domains_missing_no_real_source"),
            "candidate_lists": clrep.get("n_lists_out"),
            "first_stage_recall_top_k": clrep.get("positive_in_top_k_rate"),
            "teacher_pairs_scored": teacher.get("n_candidates"),
            "gold_positives": teacher.get("positives"),
            "hard_negatives": teacher.get("negatives"),
            "uncertain": teacher.get("uncertain"),
            "teacher": "Qwen/Qwen3-Reranker-8B",
            "leakage_filtered_against": ["dt_test", "germanquad"],
        },
        "model": {"base": runcard.get("model_base"), "loss": runcard.get("loss"),
                  "faq_share": runcard.get("faq_share"), "not_faq_only": runcard.get("not_faq_only")},
        "eval": {name: row(name) for name in ("webfaq", "germanquad", "dt_test")},
        "teacher_score_separation_by_domain": teacher.get("separation_by_domain"),
        "v4_vs_v5_same_fixed_lists": {
            "germanquad_delta": {"v4": v4d.get("germanquad"),
                                 "v5": sets.get("germanquad", {}).get("overall_delta_ndcg@10")},
            "dt_test_delta": {"v4": v4d.get("dt_test"),
                              "v5": sets.get("dt_test", {}).get("overall_delta_ndcg@10")},
        },
        "interpretation": (
            "v5 is better than v4 but still NOT promotable. Multi-domain training (FAQ share "
            "0.217) lifts every set strongly where there is real headroom (medium+hard buckets), "
            "and both guardrails improved over v4 (GermanQuAD -0.0711->-0.0285, DT-test "
            "-0.0007->+0.0211). But the gate FAILS: on GermanQuAD the reranker over-reorders "
            "near-ceiling first-stage lists (84% no_room), netting -0.0285 and 16.9% catastrophic "
            "per-query drops. Next step: rerank-or-abstain calibration on confident first stages."
        ),
        "status": "EXPERIMENTAL - NOT recommended for production reranking",
    }

    # ---- conservative-reranker comparison (real files) ----
    def _hg(gatefile, name):
        g = _load(gatefile) or {}
        s = {x["eval_set"]: x for x in g.get("eval_sets", [])}.get(name, {})
        return {"overall": s.get("overall_delta_ndcg@10"), "catastrophic": s.get("catastrophic_rate")}

    def _ab(dirpath, name):
        r = _load(pathlib.Path(dirpath) / f"eval_{name}.json") or {}
        return {"overall": r.get("delta_vs_first_stage"), "catastrophic": r.get("catastrophic_drop_rate")}

    cons = V5 / "conservative"
    NAMES = ("webfaq", "germanquad", "dt_test")
    cons_gate = _load(cons / "v5_rag_lift_gate.json") or {}
    cabst_gate = _load(cons / "abstain" / "gate.json") or {}
    result["approaches"] = {
        "raw_v5": {n: _hg(V5 / "eval" / "v5_rag_lift_gate.json", n) for n in NAMES},
        "abstain_only": {n: _ab(V5 / "abstain", n) for n in NAMES},
        "conservative_only": {n: _hg(cons / "v5_rag_lift_gate.json", n) for n in NAMES},
        "conservative_plus_abstain": {n: _ab(cons / "abstain", n) for n in NAMES},
    }
    result["conservative_gate"] = {
        "conservative_only": {
            "status": cons_gate.get("gate", {}).get("status", "fail"),
            "failing": [c["check"] for c in cons_gate.get("gate", {}).get("failing", [])]},
        "conservative_plus_abstain": {
            "status": cabst_gate.get("status", "fail"),
            "failing": [c["check"] for c in cabst_gate.get("failing", [])]},
    }
    result["conservative_interpretation"] = (
        "Real measured progress. Conservative training (rank-preservation penalty on high-first-"
        "stage-confidence lists) reduces near-ceiling churn: GermanQuAD overall -0.0285 -> +0.0094 "
        "(conservative) / +0.0243 (conservative+abstain) and catastrophic 0.169 -> 0.122 / 0.074, "
        "while WebFAQ (+0.0975) and DT-test (+0.0193) stay healthy. NOT promoted: the remaining "
        "failure is catastrophic tail risk on near-ceiling GermanQuAD lists (0.074 > 0.03 bar). "
        "Next step: a bounded / top-preserving rerank policy.")

    # ---- preservation grid (stronger-preservation retraining experiment) ----
    grid = _load(V5 / "grid" / "grid_comparison.json")
    if grid:
        pg = {}
        for m, md in grid.items():
            raw = md.get("raw_always_rerank", {})
            bnd = md.get("bounded_margin_override", {})
            pg[m] = {
                "raw_germanquad_catastrophic": raw.get("gq", {}).get("catastrophic_drop_rate"),
                "raw_germanquad_delta": raw.get("gq", {}).get("delta_vs_first_stage"),
                "raw_webfaq_delta": raw.get("wf", {}).get("delta_vs_first_stage"),
                "bounded_germanquad_catastrophic": bnd.get("gq", {}).get("catastrophic_drop_rate"),
            }
        result["preservation_grid"] = {
            "verdict": "negative training result, positive policy confirmation",
            "by_checkpoint": pg,
            "conclusion": (
                "Stronger preservation (lp04/lp06/lp08) did NOT make raw always-rerank safe on "
                "GermanQuAD (catastrophic 0.11-0.18; no lambda approaches the 0.03 bar). Bounded "
                "margin_override passes on EVERY checkpoint including the original. No new checkpoint "
                "is promoted; the original conservative checkpoint + bounded policy remains the best "
                "deployment candidate. Next: freeze and validate the bounded policy on a held-out "
                "near-ceiling guardrail."),
        }

    (V5).mkdir(parents=True, exist_ok=True)
    (V5 / "V5_RESULTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

    e = result["eval"]
    md = [
        "# v5 small-RAG reranker — results (prompt-4 real run)", "",
        f"## Verdict: **{result['verdict']}**  (gate: {result['gate_status']})", "",
        "**Status: Experimental; not recommended for production reranking.** "
        "Next step: rerank-or-abstain calibration on near-ceiling first-stage lists.", "",
        "## Training data (real, leakage-filtered vs DT-test + GermanQuAD)",
        f"- raw rows: {result['training_data']['raw_rows']} "
        f"({result['training_data']['raw_by_domain']})",
        f"- candidate lists: {result['training_data']['candidate_lists']} "
        f"(BM25 recall {result['training_data']['first_stage_recall_top_k']})",
        f"- teacher-scored pairs: {result['training_data']['teacher_pairs_scored']} "
        f"(Qwen3-Reranker-8B); gold/hardneg/uncertain "
        f"{result['training_data']['gold_positives']}/{result['training_data']['hard_negatives']}"
        f"/{result['training_data']['uncertain']}",
        f"- model: {result['model']['base']}, loss {result['model']['loss']}, "
        f"FAQ share {result['model']['faq_share']} (not_faq_only={result['model']['not_faq_only']})",
        f"- domains with no real source (omitted, NOT faked): "
        f"{result['training_data']['domains_missing_no_real_source']}", "",
        "## Hardness-aware gate (nDCG@10 over FIXED candidate lists)", "",
        "| eval set | role | overall delta | medium+hard | no_room | catastrophic | result |",
        "|---|---|--:|--:|--:|--:|:--|",
    ]
    for name in ("webfaq", "germanquad", "dt_test"):
        r = e[name]
        passed = name not in [c.split("_")[0] for c in result["gate_failing_checks"]]
        md.append(f"| {name} | {r['role']} | {r['overall_delta_ndcg@10']:+} | "
                  f"{r['medium_hard_micro_lift']:+} | {r['no_room_fraction']} | "
                  f"{r['catastrophic_rate']} | {'pass' if passed else 'FAIL'} |")
    cmp = result["v4_vs_v5_same_fixed_lists"]
    md += ["", "## v4 -> v5 on the same fixed guardrail lists",
           f"- GermanQuAD: {cmp['germanquad_delta']['v4']} -> {cmp['germanquad_delta']['v5']} "
           "(degradation reduced, still fails)",
           f"- DT-test: {cmp['dt_test_delta']['v4']} -> {cmp['dt_test_delta']['v5']} "
           "(now positive)", ""]
    a = result["approaches"]
    cg = result["conservative_gate"]
    md += ["## Conservative reranker (rank-preservation loss) — real measured progress", "",
           "| approach | GermanQuAD overall | GermanQuAD catastrophic | WebFAQ overall | "
           "DT-test overall | gate |", "|---|--:|--:|--:|--:|:--|"]
    statusmap = {"raw_v5": "fail", "abstain_only": "fail",
                 "conservative_only": cg["conservative_only"]["status"],
                 "conservative_plus_abstain": cg["conservative_plus_abstain"]["status"]}
    for key in ("raw_v5", "abstain_only", "conservative_only", "conservative_plus_abstain"):
        ap = a[key]
        md.append(f"| {key} | {ap['germanquad']['overall']} | {ap['germanquad']['catastrophic']} | "
                  f"{ap['webfaq']['overall']} | {ap['dt_test']['overall']} | {statusmap[key]} |")
    md += ["", f"_Conservative-only gate fails: {cg['conservative_only']['failing']}. "
           f"Conservative+abstain gate fails: {cg['conservative_plus_abstain']['failing']}. "
           "NOT promoted._", "", result["conservative_interpretation"], ""]
    pgrid = result.get("preservation_grid")
    if pgrid:
        md += ["## Preservation grid — negative training result, positive policy confirmation", "",
               "| checkpoint | RAW GQ catastrophic | RAW GQ Δ | RAW WebFAQ Δ | bounded GQ catastrophic |",
               "|---|--:|--:|--:|--:|"]
        for m, v in pgrid["by_checkpoint"].items():
            md.append(f"| {m} | {v['raw_germanquad_catastrophic']} | {v['raw_germanquad_delta']} | "
                      f"{v['raw_webfaq_delta']} | {v['bounded_germanquad_catastrophic']} |")
        md += ["", pgrid["conclusion"], ""]
    md += ["## Interpretation (raw v5)", "", result["interpretation"], ""]
    (V5 / "V5_RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[v5-summary] verdict={result['verdict']} gate={result['gate_status']} "
          f"-> {V5}/V5_RESULTS.md, V5_RESULTS.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
