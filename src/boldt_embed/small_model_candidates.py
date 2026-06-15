"""v5 small-model candidate comparison (pure stdlib core, no ML).

Decides whether Boldt is actually the best *small* base for the German RAG job, or whether
`Qwen/Qwen3-Embedding-0.6B` / `Qwen/Qwen3-Reranker-0.6B` (also small, multilingual, instruction-
aware, 32k context, already trained for embedding/reranking) win. The production default is chosen
by **quality then latency — family-blind** — never "because it's Boldt" or "because it's Qwen".

This module holds the testable logic: config validation, the storage/params helpers, and the
selection gate. The ML measurement (loading models, timing, VRAM) lives in
`scripts/eval_small_model_candidates.py` behind lazy imports; dry-run imports no ML.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

DEFAULT_TIE_BREAK_QUALITY_DELTA = 0.005    # within this nDCG@10, prefer the faster model
DEFAULT_MIN_256D_RETENTION = 0.95


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _validate_candidate(c: Any, idx: int, kind: str) -> List[str]:
    errs: List[str] = []
    if not isinstance(c, dict):
        return [f"{kind}_candidates[{idx}]: not an object"]
    for k in ("name", "model_name_or_path", "family", "backend"):
        if not _nonempty_str(c.get(k)):
            errs.append(f"{kind}_candidates[{idx}] ({c.get('name', '?')}): '{k}' must be a non-empty string")
    return errs


def validate_candidates_config(d: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    if not _nonempty_str(d.get("experiment_id")):
        errs.append("'experiment_id' must be a non-empty string")
    if not _nonempty_str(d.get("goal")):
        errs.append("'goal' must be a non-empty string")

    for kind in ("dense", "reranker"):
        lst = d.get(f"{kind}_candidates")
        if not isinstance(lst, list) or not lst:
            errs.append(f"'{kind}_candidates' must be a non-empty list")
        else:
            for i, c in enumerate(lst):
                errs += _validate_candidate(c, i, kind)
            fams = {c.get("family") for c in lst if isinstance(c, dict)}
            if len(fams) < 2:
                errs.append(f"'{kind}_candidates' must include >= 2 model families for a real "
                            f"comparison (got {sorted(f for f in fams if f)})")

    teachers = d.get("teachers")
    if not isinstance(teachers, dict):
        errs.append("'teachers' must be an object with 'embedding' and 'reranker'")
    else:
        for k in ("embedding", "reranker"):
            if not _nonempty_str(teachers.get(k)):
                errs.append(f"teachers.{k} must be a non-empty string")

    tuning = d.get("tuning")
    if not isinstance(tuning, dict):
        errs.append("'tuning' must be an object")
    else:
        if tuning.get("method") not in ("lora", "full"):
            errs.append("tuning.method must be 'lora' or 'full'")
        if tuning.get("method") == "full" and tuning.get("full_finetune_allowed") is not True:
            errs.append("tuning.method 'full' requires tuning.full_finetune_allowed = true")

    gate = d.get("selection_gate")
    if not isinstance(gate, dict):
        errs.append("'selection_gate' must be an object")
    else:
        for k in ("max_reranker_latency_ms", "max_dense_latency_ms"):
            if not _is_number(gate.get(k)) or gate[k] <= 0:
                errs.append(f"selection_gate.{k} must be a positive number")
        tb = gate.get("tie_break_quality_delta", DEFAULT_TIE_BREAK_QUALITY_DELTA)
        if not _is_number(tb) or tb < 0:
            errs.append("selection_gate.tie_break_quality_delta must be >= 0")
        ret = gate.get("min_256d_retention", DEFAULT_MIN_256D_RETENTION)
        if not _is_number(ret) or not (0 < ret <= 1):
            errs.append("selection_gate.min_256d_retention must be in (0, 1]")
    return errs


def storage_table(dims: List[int]) -> Dict[str, Dict[str, int]]:
    """Bytes per vector at each embedding dim, fp32 and fp16."""
    return {str(d): {"fp32_bytes": d * 4, "fp16_bytes": d * 2} for d in dims}


def select_default(results: List[Dict[str, Any]], *, max_latency_ms: float,
                   tie_break_quality_delta: float = DEFAULT_TIE_BREAK_QUALITY_DELTA,
                   min_256d_retention: Optional[float] = None) -> Dict[str, Any]:
    """Pick the production default by quality-then-latency, FAMILY-BLIND, over same-harness results.

    ``results``: list of measured candidates, each ``{name, family, quality, latency_ms, harness,
    [retention_256d]}``. Refuses to choose unless there are >= 2 candidates measured under the
    SAME harness (acceptance: no model is promoted without a same-harness comparison)."""
    if len(results) < 2:
        return {"status": "insufficient_comparison", "default": None,
                "reason": "need >= 2 candidates measured under the same harness"}
    harnesses = {r.get("harness") for r in results}
    if len(harnesses) != 1 or None in harnesses:
        return {"status": "inconsistent_harness", "default": None,
                "reason": f"candidates not measured under one harness: {sorted(map(str, harnesses))}"}

    excluded = []
    eligible = []
    for r in results:
        why = []
        if r["latency_ms"] > max_latency_ms:
            why.append(f"latency {r['latency_ms']}ms > {max_latency_ms}ms")
        if min_256d_retention is not None and r.get("retention_256d", 1.0) < min_256d_retention:
            why.append(f"256d retention {r.get('retention_256d')} < {min_256d_retention}")
        (excluded if why else eligible).append({"name": r["name"], "reason": "; ".join(why)} if why else r)

    if not eligible:
        return {"status": "no_eligible_candidate", "default": None,
                "reason": "no candidate met latency/retention budget", "excluded": excluded}

    best_q = max(r["quality"] for r in eligible)
    # family-blind: among candidates within tie-break quality of the best, take the fastest.
    contenders = [r for r in eligible if best_q - r["quality"] <= tie_break_quality_delta + 1e-9]
    default = min(contenders, key=lambda r: (r["latency_ms"], -r["quality"], r["name"]))
    ranking = sorted(eligible, key=lambda r: (-r["quality"], r["latency_ms"], r["name"]))
    return {
        "status": "selected",
        "default": default["name"],
        "default_family": default["family"],
        "selected_by": "quality_then_latency (family-blind)",
        "best_quality": round(best_q, 6),
        "ranking": [{"name": r["name"], "family": r["family"], "quality": r["quality"],
                     "latency_ms": r["latency_ms"]} for r in ranking],
        "excluded": excluded,
    }


def measurement_plan(config: Dict[str, Any], *, tune_reranker: bool, tune_embedding: bool,
                     full_finetune: bool) -> Dict[str, Any]:
    """Dry-run plan: which candidates/modes run, and the gate — no measurement, no ML."""
    method = "full" if full_finetune else "lora"
    return {
        "experiment_id": config.get("experiment_id"),
        "dense_candidates": [c["name"] for c in config.get("dense_candidates", [])],
        "reranker_candidates": [c["name"] for c in config.get("reranker_candidates", [])],
        "families": {
            "dense": sorted({c["family"] for c in config.get("dense_candidates", [])}),
            "reranker": sorted({c["family"] for c in config.get("reranker_candidates", [])}),
        },
        "tuning": {"reranker_lora": tune_reranker, "embedding_lora": tune_embedding,
                   "method": method, "teacher": config.get("teachers", {}),
                   "full_finetune": full_finetune},
        "selection_gate": config.get("selection_gate", {}),
        "reported_metrics": ["quality", "latency_ms", "vram_mb", "params_m", "throughput_qps",
                             "storage_bytes_per_vector"],
        "note": "dry-run plan only; real run measures all candidates under the SAME harness, "
                "then select_default() picks the production default by quality-then-latency.",
    }
