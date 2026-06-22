#!/usr/bin/env python3
"""Canonical AutoResearch scorer for dense-retriever trials (pure stdlib, deterministic).

PROTECTED SURFACE: future AutoResearch experiments must NOT edit this file except by intentional
human review. It defines the weighted score and the fail-closed hard gates that decide whether a
trial is promotable.

Score (deltas are run − baseline):

    score =
      + 2.0 * Δwebfaq_recall@100
      + 1.5 * Δwebfaq_ndcg@10
      + 1.0 * Δlocal_rag_recall@100   (only when both run and baseline have local_rag)
      + 0.5 * Δwebfaq_mrr@10
      - 3.0 * germanquad_regression_penalty
      - 3.0 * dt_test_regression_penalty
      - 2.0 * matryoshka_256_retention_penalty
      - 0.2 * vram_penalty
      - 0.2 * throughput_penalty

Hard gates (status "pass" only if ALL hold): run status ok/pass; leakage hits ≤ max; ΔGermanQuAD
nDCG@10 ≥ min_delta; ΔDT-test nDCG@10 ≥ min_delta; 256-d retention ≥ min;
WebFAQ recall@100 and nDCG@10 present.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

DEFAULTS = {
    "germanquad_min_delta": -0.005,
    "dt_test_min_delta": -0.005,
    "m256_min_retention": 0.95,
    "max_leakage_hits": 0,
}

# Defensive aliases for metric names (matched case-insensitively).
_ALIASES = {
    "recall@100": ["recall@100", "recall_at_100", "recall100"],
    "ndcg@10": ["ndcg@10", "ndcg_at_10", "ndcg10"],
    "mrr@10": ["mrr@10", "mrr_at_10", "mrr10"],
    "retention_256": ["retention_256", "retention256", "m256_retention",
                      "matryoshka_256_retention"],
}
_LEAKAGE_KEYS = ["hits", "num_hits", "leakage_hits"]


def _get(d: Optional[Dict[str, Any]], canonical: str) -> Optional[float]:
    """Look up a metric value by canonical name, tolerating known aliases / casing."""
    if not isinstance(d, dict):
        return None
    lower = {str(k).lower(): v for k, v in d.items()}
    for cand in _ALIASES.get(canonical, [canonical]):
        v = lower.get(cand.lower())
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def _metrics_block(doc: Dict[str, Any]) -> Dict[str, Any]:
    m = doc.get("metrics")
    return m if isinstance(m, dict) else {}


# Leakage statuses that mean "not verified clean" — these fail the gate regardless of hit count.
_LEAKAGE_BAD_STATUSES = {"not_checked", "unparseable", "unreadable", "missing_report",
                         "leak_detected", "unknown"}


def _leakage(doc: Dict[str, Any]):
    """Return (hits, status) from the leakage block. hits is None when no count is present —
    which the gate treats as 'not verified', NOT as zero."""
    block = _metrics_block(doc).get("leakage")
    if not isinstance(block, dict):
        return None, None
    lower = {str(k).lower(): v for k, v in block.items()}
    hits = None
    for key in _LEAKAGE_KEYS:
        v = lower.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            hits = int(v)
            break
    if hits is None:
        summ = block.get("summary")
        if isinstance(summ, dict) and isinstance(summ.get("hits"), (int, float)):
            hits = int(summ["hits"])
    status = block.get("status")
    return hits, (str(status).lower() if status is not None else None)


def _delta(run_v: Optional[float], base_v: Optional[float]) -> Optional[float]:
    if run_v is None:
        return None
    if base_v is None:
        return None
    return run_v - base_v


def score_run(run: Dict[str, Any], baseline: Dict[str, Any],
              thresholds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compute the weighted score, deltas, penalties, and fail-closed gates. Pure function."""
    th = dict(DEFAULTS)
    if thresholds:
        th.update({k: v for k, v in thresholds.items() if v is not None})

    rm, bm = _metrics_block(run), _metrics_block(baseline)
    r_wf, b_wf = rm.get("webfaq"), bm.get("webfaq")
    r_gq, b_gq = rm.get("germanquad"), bm.get("germanquad")
    r_dt, b_dt = rm.get("dt_test"), bm.get("dt_test")
    r_lr, b_lr = rm.get("local_rag"), bm.get("local_rag")
    r_mat = rm.get("matryoshka")
    r_sys, b_sys = rm.get("system") or {}, bm.get("system") or {}

    # --- raw values ---
    run_wf_recall = _get(r_wf, "recall@100")
    run_wf_ndcg = _get(r_wf, "ndcg@10")
    retention_256 = _get(r_mat, "retention_256")

    # --- deltas ---
    deltas: Dict[str, Any] = {
        "webfaq_recall@100": _delta(run_wf_recall, _get(b_wf, "recall@100")),
        "webfaq_ndcg@10": _delta(run_wf_ndcg, _get(b_wf, "ndcg@10")),
        "webfaq_mrr@10": _delta(_get(r_wf, "mrr@10"), _get(b_wf, "mrr@10")),
        "germanquad_ndcg@10": _delta(_get(r_gq, "ndcg@10"), _get(b_gq, "ndcg@10")),
        "dt_test_ndcg@10": _delta(_get(r_dt, "ndcg@10"), _get(b_dt, "ndcg@10")),
    }
    has_local_rag = isinstance(r_lr, dict) and isinstance(b_lr, dict)
    if has_local_rag:
        deltas["local_rag_recall@100"] = _delta(_get(r_lr, "recall@100"),
                                                _get(b_lr, "recall@100"))

    # --- penalties (positive magnitudes only) ---
    def pos(x: Optional[float]) -> float:
        return max(0.0, x) if x is not None else 0.0

    d_gq, d_dt = deltas["germanquad_ndcg@10"], deltas["dt_test_ndcg@10"]
    run_vram, base_vram = r_sys.get("vram_gb"), b_sys.get("vram_gb")
    run_tput = r_sys.get("throughput_pairs_per_sec")
    base_tput = b_sys.get("throughput_pairs_per_sec")

    gq_short = (th["germanquad_min_delta"] - d_gq) if d_gq is not None else None
    penalties: Dict[str, float] = {
        "germanquad_regression": pos(gq_short),
        "dt_test_regression": pos((th["dt_test_min_delta"] - d_dt) if d_dt is not None else None),
        "matryoshka_256_retention": pos(
            (th["m256_min_retention"] - retention_256) if retention_256 is not None else None),
        "vram": (max(0.0, (run_vram - base_vram)) / max(1.0, base_vram))
        if isinstance(run_vram, (int, float)) and isinstance(base_vram, (int, float)) else 0.0,
        "throughput": (max(0.0, (base_tput - run_tput)) / max(1.0, base_tput))
        if isinstance(run_tput, (int, float)) and isinstance(base_tput, (int, float)) else 0.0,
    }

    def term(d: Optional[float], w: float) -> float:
        return w * d if d is not None else 0.0

    score = (
        term(deltas["webfaq_recall@100"], 2.0)
        + term(deltas["webfaq_ndcg@10"], 1.5)
        + term(deltas.get("local_rag_recall@100"), 1.0)  # 0 when local_rag absent (key unset)
        + term(deltas["webfaq_mrr@10"], 0.5)
        - 3.0 * penalties["germanquad_regression"]
        - 3.0 * penalties["dt_test_regression"]
        - 2.0 * penalties["matryoshka_256_retention"]
        - 0.2 * penalties["vram"]
        - 0.2 * penalties["throughput"]
    )

    # --- hard gates (fail-closed) ---
    failed: List[Dict[str, Any]] = []
    run_status = str(run.get("status", "ok")).lower()
    if run_status not in ("ok", "pass"):
        failed.append({"name": "run_status", "value": run_status, "threshold": "ok|pass"})
    # A dry-run / non-real trial carries plumbing-only pseudo-metrics — never promotable.
    mode = str(run.get("mode") or "").lower()
    if (mode and mode != "real") or run.get("scale_disclaimer"):
        failed.append({"name": "not_a_real_run", "threshold": "real",
                       "value": run.get("mode") or "scale_disclaimer set"})
    # The comparison is only meaningful against a real measured baseline (not the 0.0 skeleton).
    base_wf_recall = _get(b_wf, "recall@100")
    if base_wf_recall is None or base_wf_recall <= 0.0:
        failed.append({"name": "baseline_incomplete",
                       "value": {"webfaq_recall@100": base_wf_recall},
                       "threshold": "real measured baseline (webfaq recall@100 > 0)"})
    elif _get(b_wf, "ndcg@10") is None:
        failed.append({"name": "baseline_incomplete", "value": {"webfaq_ndcg@10": None},
                       "threshold": "real measured baseline (webfaq ndcg@10 present)"})
    # Leakage must be VERIFIED clean — a missing/unchecked block fails closed (not treated as 0).
    leak_hits, leak_status = _leakage(run)
    if (leak_hits is None or leak_status in _LEAKAGE_BAD_STATUSES
            or leak_hits > th["max_leakage_hits"]):
        failed.append({"name": "leakage", "value": {"hits": leak_hits, "status": leak_status},
                       "threshold": f"verified clean, hits <= {th['max_leakage_hits']}"})
    if d_gq is None or d_gq < th["germanquad_min_delta"] - 1e-12:
        failed.append({"name": "germanquad_ndcg@10_delta", "value": d_gq,
                       "threshold": th["germanquad_min_delta"]})
    if d_dt is None or d_dt < th["dt_test_min_delta"] - 1e-12:
        failed.append({"name": "dt_test_ndcg@10_delta", "value": d_dt,
                       "threshold": th["dt_test_min_delta"]})
    if retention_256 is None or retention_256 < th["m256_min_retention"] - 1e-12:
        failed.append({"name": "matryoshka_256_retention", "value": retention_256,
                       "threshold": th["m256_min_retention"]})
    if run_wf_recall is None:
        failed.append({"name": "webfaq_recall@100_present", "value": None, "threshold": "present"})
    if run_wf_ndcg is None:
        failed.append({"name": "webfaq_ndcg@10_present", "value": None, "threshold": "present"})

    return {
        "status": "pass" if not failed else "fail",
        "score": round(score, 6),
        "deltas": deltas,
        "penalties": {k: round(v, 6) for k, v in penalties.items()},
        "failed_gates": failed,
        "thresholds": th,
        "has_local_rag": has_local_rag,
    }


def _to_markdown(result: Dict[str, Any], run_path: str, baseline_path: str) -> str:
    lines = [f"# AutoResearch score: **{result['status']}**", "",
             f"- score: `{result['score']}`",
             f"- run: `{run_path}`", f"- baseline: `{baseline_path}`", "",
             "| delta | value |", "|---|---:|"]
    for k, v in result["deltas"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "| penalty | value |", "|---|---:|"]
    for k, v in result["penalties"].items():
        lines.append(f"| {k} | {v} |")
    if result["failed_gates"]:
        lines += ["", "## Failed gates"]
        for g in result["failed_gates"]:
            lines.append(f"- **{g['name']}**: value `{g['value']}` vs threshold `{g['threshold']}`")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=["json", "markdown"], default="json")
    ap.add_argument("--germanquad-min-delta", type=float, default=DEFAULTS["germanquad_min_delta"])
    ap.add_argument("--dt-test-min-delta", type=float, default=DEFAULTS["dt_test_min_delta"])
    ap.add_argument("--m256-min-retention", type=float, default=DEFAULTS["m256_min_retention"])
    ap.add_argument("--max-leakage-hits", type=int, default=DEFAULTS["max_leakage_hits"])
    args = ap.parse_args(argv)

    run = json.loads(pathlib.Path(args.run).read_text(encoding="utf-8"))
    baseline = json.loads(pathlib.Path(args.baseline).read_text(encoding="utf-8"))
    result = score_run(run, baseline, {
        "germanquad_min_delta": args.germanquad_min_delta,
        "dt_test_min_delta": args.dt_test_min_delta,
        "m256_min_retention": args.m256_min_retention,
        "max_leakage_hits": args.max_leakage_hits,
    })
    result["inputs"] = {"run": args.run, "baseline": args.baseline}

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.format == "markdown":
        print(_to_markdown(result, args.run, args.baseline))
    else:
        print(json.dumps({"status": result["status"], "score": result["score"],
                          "failed_gates": [g["name"] for g in result["failed_gates"]]},
                         ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
