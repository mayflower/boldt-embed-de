#!/usr/bin/env python3
"""v5 small-model candidate orchestrator: compare Boldt vs Qwen3-0.6B (and other small baselines)
under ONE harness, optionally LoRA-tune the Qwen3-0.6B reranker/embedder, and choose the production
default by quality-then-latency — family-blind.

Modes (compose):
  (default)            baseline-only evaluation of every dense + reranker candidate
  --tune-reranker      LoRA-tune the Qwen3-Reranker-0.6B on v5 teacher-scored lists, then eval it
  --tune-embedding     LoRA-tune the Qwen3-Embedding-0.6B, then eval it
  --full-finetune      allow full fine-tune instead of LoRA (otherwise LoRA only)

Dry-run imports NO ML and emits the plan (candidates, modes, gate). A real run measures quality,
latency, VRAM, params, throughput, and storage-at-embedding-dims for every candidate, then runs the
selection gate. No model is promoted without a same-harness comparison (>= 2 candidates).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import small_model_candidates as SMC  # noqa: E402


def _read(path: pathlib.Path) -> list:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").split("\n") if ln.strip()]


def _harness_id(parts) -> str:
    import hashlib
    return "h" + hashlib.blake2b("|".join(str(p) for p in parts).encode("utf-8"),
                                 digest_size=6).hexdigest()


def _measure_real(config, args, harness):
    """Real-run measurement (lazy ML). Returns (dense_results, reranker_results, errors)."""
    from boldt_embed import small_model_measure as MEAS  # lazy ML helpers
    dense, rerank, errors = [], [], []
    if args.dense_eval:
        eval_set = _read(pathlib.Path(args.dense_eval))
        corpus = _read(pathlib.Path(args.dense_corpus)) if args.dense_corpus else []
        for c in config["dense_candidates"]:
            try:
                m = MEAS.measure_dense(c, eval_set, corpus, args.device)
                m["harness"] = harness
                dense.append(m)
            except Exception as exc:  # noqa: BLE001 — record, don't abort the bake-off
                errors.append(f"dense {c['name']}: {exc}")
    if args.reranker_eval:
        lists = _read(pathlib.Path(args.reranker_eval))
        for c in config["reranker_candidates"]:
            try:
                m = MEAS.measure_reranker(c, lists, args.device)
                m["harness"] = harness
                rerank.append(m)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"reranker {c['name']}: {exc}")
    return dense, rerank, errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "v5_small_model_candidates.json"))
    ap.add_argument("--dense-eval", default=None, help="dense retrieval eval set JSONL (real run)")
    ap.add_argument("--dense-corpus", default=None, help="dense corpus JSONL (real run)")
    ap.add_argument("--reranker-eval", default=None, help="FIXED candidate lists JSONL (real run)")
    ap.add_argument("--tune-reranker", action="store_true")
    ap.add_argument("--tune-embedding", action="store_true")
    ap.add_argument("--full-finetune", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report", default="outputs/v5-small-rag/small_model_candidates_report.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    cfg_errors = SMC.validate_candidates_config(config)
    if cfg_errors:
        print(f"ERROR: invalid candidates config: {cfg_errors[0]}", file=sys.stderr)
        return 2
    if args.full_finetune and config.get("tuning", {}).get("full_finetune_allowed") is not True:
        print("ERROR: --full-finetune but config tuning.full_finetune_allowed != true", file=sys.stderr)
        return 2

    gate = config["selection_gate"]
    out_report: dict = {"experiment_id": config["experiment_id"], "selection_gate": gate}

    if args.dry_run:
        plan = SMC.measurement_plan(config, tune_reranker=args.tune_reranker,
                                    tune_embedding=args.tune_embedding,
                                    full_finetune=args.full_finetune)
        plan["storage_at_dims"] = {c["name"]: SMC.storage_table(c.get("embedding_dims", []))
                                   for c in config["dense_candidates"]}
        out_report["plan"] = plan
        assert "torch" not in sys.modules, "dry-run must not import torch"
        _write_report(args.report, out_report)
        print(f"[v5-candidates] dry-run plan: dense={plan['dense_candidates']} "
              f"reranker={plan['reranker_candidates']} tuning={plan['tuning']['method']} "
              f"-> {args.report}")
        return 0

    # ---- real run (lazy ML) ----
    harness = _harness_id([args.config, args.dense_eval, args.dense_corpus, args.reranker_eval,
                           args.tune_reranker, args.tune_embedding, args.full_finetune])
    if args.tune_reranker or args.tune_embedding:
        from boldt_embed import small_model_measure as MEAS  # lazy ML
        config = MEAS.lora_tune_candidates(config, args)     # appends tuned candidate(s)

    dense, rerank, errors = _measure_real(config, args, harness)
    out_report["harness"] = harness
    out_report["errors"] = errors
    out_report["dense_results"] = dense
    out_report["reranker_results"] = rerank
    out_report["dense_selection"] = SMC.select_default(
        dense, max_latency_ms=gate["max_dense_latency_ms"],
        tie_break_quality_delta=gate.get("tie_break_quality_delta", 0.005),
        min_256d_retention=gate.get("min_256d_retention", 0.95)) if dense else {"status": "not_run"}
    out_report["reranker_selection"] = SMC.select_default(
        rerank, max_latency_ms=gate["max_reranker_latency_ms"],
        tie_break_quality_delta=gate.get("tie_break_quality_delta", 0.005)) if rerank else {"status": "not_run"}
    _write_report(args.report, out_report)

    print(f"[v5-candidates] dense default={out_report['dense_selection'].get('default')} "
          f"reranker default={out_report['reranker_selection'].get('default')} "
          f"(family-blind) -> {args.report}")
    for e in errors:
        print(f"  ✗ {e}", file=sys.stderr)
    ok = (out_report["dense_selection"].get("status") in ("selected", "not_run")
          and out_report["reranker_selection"].get("status") in ("selected", "not_run")
          and not (out_report["dense_selection"].get("status") == "not_run"
                   and out_report["reranker_selection"].get("status") == "not_run"))
    return 0 if ok else 1


def _write_report(path, report):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
