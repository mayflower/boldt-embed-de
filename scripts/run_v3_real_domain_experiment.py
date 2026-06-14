#!/usr/bin/env python3
"""Reproducible v3 real-domain-generalization orchestrator. SAFE by default: ``dry-run`` only
plans (no torch, no downloads); ``full`` (with --i-understand-this-runs-gpu) executes the GPU
pipeline. It orchestrates the existing v3 scripts — it does not duplicate their logic — and is
built so a run with unknown licenses, a missing leakage scan, capped mining, or reranker
degradation CANNOT be silently promoted.

Writes COMMANDS.md + STATUS.json + V3_RESULTS.{md,json} under --work-dir.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import domain_source_acquisition as dsa  # noqa: E402
from boldt_embed.v3_experiment_config import load_v3_experiment_config  # noqa: E402

PY = sys.executable
S = str(ROOT / "scripts")
EVAL = str(ROOT / "data" / "processed" / "eval")
EVAL_CORPORA = [f"{EVAL}/germanquad_corpus.jsonl", f"{EVAL}/germanquad_queries.jsonl",
                f"{EVAL}/dt_test_corpus.jsonl", f"{EVAL}/dt_test_queries.jsonl"]


def _stages(args, w):
    """Ordered stages: {name, argv, gpu, gate}. ``gate`` stages can be bypassed with
    --allow-research-failures (verdict becomes invalid_for_promotion); others always stop."""
    tc = f"{w}/teacher-cache/qwen3_v3"
    cand = f"{w}/candidates_v3.jsonl"
    clean = f"{w}/candidates_v3.clean.jsonl"
    idx = f"{w}/leakage/eval_index.json"
    leak_report = f"{w}/leakage/leakage_report.json"
    filt_emb = f"{tc}.filtered_embedder.jsonl"
    filt_rr = f"{tc}.filtered_reranker.jsonl"
    bm25 = f"{w}/bm25_v3.json"
    acquire_mode = "materialize-local" if args.mode == "full" else "dry-run"
    st = []

    st.append({"name": "acquire_sources", "gpu": False, "gate": False, "argv": [
        PY, f"{S}/acquire_v3_sources.py", "--manifest", args.manifest,
        "--output-dir", "data/raw/v3", "--mode", acquire_mode]})
    st.append({"name": "build_leakage_index", "gpu": False, "gate": False, "argv": [
        PY, f"{S}/build_leakage_index.py", "--eval-corpus", *EVAL_CORPORA, "--output", idx]})
    st.append({"name": "build_candidates", "gpu": False, "gate": False, "argv": [
        PY, f"{S}/build_v3_candidates.py", "--manifest", args.manifest, "--config", args.config,
        "--raw-dir", "data/raw/v3", "--output", cand, "--target-count", str(args.target_count),
        "--leakage-index", idx, "--pii-scan", "--fail-on-unknown-license",
        "--fail-on-domain-quota-miss"]})
    st.append({"name": "full_leakage_scan", "gpu": False, "gate": False, "argv": [
        PY, f"{S}/run_full_leakage_scan.py", "--candidates", cand, "--eval-corpus", *EVAL_CORPORA,
        "--output", leak_report, "--hits-output", f"{w}/leakage/leakage_hits.jsonl",
        "--drop-hits", clean]})
    st.append({"name": "build_bm25_index", "gpu": False, "gate": False, "argv": [
        PY, f"{S}/build_bm25_index.py", "--corpus", clean, "--output", bm25]})
    st.append({"name": "teacher_cache", "gpu": True, "gate": False, "argv": [
        PY, f"{S}/build_teacher_cache.py", "--input", clean, "--output", f"{tc}.jsonl",
        "--mode", "both", "--shard-size", "5000", "--max-length", "512", "--resume"]})
    st.append({"name": "summarize_cache", "gpu": False, "gate": False, "argv": [
        PY, f"{S}/summarize_teacher_cache.py", "--input", f"{tc}.manifest.json",
        "--output", f"{tc}.summary.json", "--fail-on-unknown-license",
        "--fail-on-disallowed-training-source"]})
    st.append({"name": "calibrate_thresholds", "gpu": False, "gate": False, "argv": [
        PY, f"{S}/calibrate_teacher_thresholds.py", "--teacher-cache", f"{tc}.manifest.json",
        "--config", args.config, "--output", f"{tc}.calibration.json",
        "--markdown", f"{tc}.calibration.md", "--embedder-output", filt_emb,
        "--reranker-output", filt_rr]})
    st.append({"name": "domain_quality_gate", "gpu": False, "gate": True, "argv": [
        PY, f"{S}/analyze_domain_quality.py", "--candidates", clean,
        "--teacher-cache", f"{tc}.manifest.json", "--config", args.config,
        "--source-manifest", args.manifest, "--output", f"{w}/domain_quality.json",
        "--markdown", f"{w}/domain_quality.md"]})
    st.append({"name": "mine_hard_negatives", "gpu": False, "gate": False, "argv": [
        PY, f"{S}/mine_hard_negatives_2026.py", "--candidates", filt_emb,
        "--teacher-cache", f"{tc}.jsonl", "--bm25-index", bm25,
        "--output", f"{w}/hardneg_v3.jsonl", "--negatives-per-query", "8",
        "--require-full-corpus"]})
    if args.train_causal:
        st.append({"name": "train_causal", "gpu": True, "gate": False, "argv": [
            PY, f"{S}/train_modern_embedder.py", "--teacher-cache", filt_emb,
            "--hard-negatives", f"{w}/hardneg_v3.jsonl", "--bidirectional", "false",
            "--use-teacher-score-distillation", "true",
            "--require-leakage-report", leak_report,
            "--output", f"{w}/checkpoints/boldt-modern-causal-v3", "--bf16",
            "--gradient-checkpointing", "--run-id", f"{args.run_id_prefix}-causal"]})
    if args.train_bi_mntp:   # default OFF — v2 causal won
        st.append({"name": "prepare_bi_mntp", "gpu": True, "gate": False, "argv": [
            PY, f"{S}/prepare_bidirectional_student.py", "--texts", f"{w}/mntp_texts_v3.jsonl",
            "--output", f"{w}/checkpoints/boldt-bi-mntp-v3", "--steps", "600", "--bf16",
            "--gradient-checkpointing", "--run-id", f"{args.run_id_prefix}-mntp"]})
        st.append({"name": "train_bi_mntp", "gpu": True, "gate": False, "argv": [
            PY, f"{S}/train_modern_embedder.py", "--base-model", f"{w}/checkpoints/boldt-bi-mntp-v3",
            "--bidirectional", "true", "--teacher-cache", filt_emb,
            "--require-leakage-report", leak_report,
            "--output", f"{w}/checkpoints/boldt-modern-bi-mntp-v3", "--bf16",
            "--gradient-checkpointing", "--run-id", f"{args.run_id_prefix}-bi-mntp"]})
    if args.train_reranker:
        st.append({"name": "build_reranker_candidates", "gpu": False, "gate": False, "argv": [
            PY, f"{S}/build_reranker_candidates_v3.py", "--teacher-cache", filt_rr,
            "--bm25-results", bm25, "--output", f"{w}/reranker_train_v3.jsonl"]})
        st.append({"name": "train_reranker", "gpu": True, "gate": False, "argv": [
            PY, f"{S}/train_modern_reranker.py", "--candidate-lists", f"{w}/reranker_train_v3.jsonl",
            "--loss", "mixed", "--pairwise-min-teacher-margin", "2.0",
            "--output", f"{w}/checkpoints/boldt-reranker-modern-v3", "--bf16",
            "--run-id", f"{args.run_id_prefix}-reranker"]})
    if args.eval:
        for ds in ("germanquad", "dt_test", "gerdalir"):
            st.append({"name": f"eval_dense_{ds}", "gpu": True, "gate": False, "argv": [
                PY, f"{S}/run_baseline_benchmarks.py", "--models", "data/processed/v3_eval_models.json",
                "--mode", "local", "--task-name", ds, "--eval-corpus", f"{EVAL}/{ds}_corpus.jsonl",
                "--eval-queries", f"{EVAL}/{ds}_queries.jsonl", "--qrels", f"{EVAL}/{ds}_qrels.jsonl",
                "--output", f"{w}/eval/dense_{ds}.json", "--run-id", f"{args.run_id_prefix}-eval-{ds}"]})
        st.append({"name": "matryoshka_sweep", "gpu": True, "gate": False, "argv": [
            PY, f"{S}/eval_matryoshka_sweep.py", "--model", f"{w}/checkpoints/boldt-modern-causal-v3",
            "--eval-corpus", f"{EVAL}/germanquad_corpus.jsonl",
            "--eval-queries", f"{EVAL}/germanquad_queries.jsonl",
            "--qrels", f"{EVAL}/germanquad_qrels.jsonl", "--dataset", "germanquad",
            "--output", f"{w}/real_matryoshka_germanquad.json", "--run-id", f"{args.run_id_prefix}-matryoshka"]})
        if args.train_reranker:
            for ds in ("germanquad", "dt_test"):
                st.append({"name": f"reranker_lift_{ds}", "gpu": True, "gate": False, "argv": [
                    PY, f"{S}/eval_reranker_lift.py", "--candidates", f"{EVAL}/{ds}_shortlist.jsonl",
                    "--reranker", f"{w}/checkpoints/boldt-reranker-modern-v3",
                    "--output", f"{w}/reranker-lift-{ds}-v3.json", "--run-id", f"{args.run_id_prefix}-lift-{ds}"]})
            st.append({"name": "reranker_promotion_gate", "gpu": False, "gate": True, "argv": [
                PY, f"{S}/check_reranker_promotion_gate.py",
                "--dt-test", f"{w}/reranker-lift-dt_test-v3.json",
                "--germanquad", f"{w}/reranker-lift-germanquad-v3.json",
                "--training-summary", f"{w}/checkpoints/boldt-reranker-modern-v3/reranker_training_summary.json",
                "--output", f"{w}/reranker_gate.json"]})
    st.append({"name": "release_gate", "gpu": False, "gate": True, "argv": [
        PY, f"{S}/validate_release_2026.py", "--require-v3-artifacts", "--results-dir", w]})
    return st


def _verdict(status_stages, mode, invalid_for_promotion):
    if any(s["status"].startswith("failed") for s in status_stages):
        return "failed"
    if invalid_for_promotion:
        return "invalid_for_promotion"
    return {"dry-run": "planned", "smoke": "smoke-ok", "full": "promotable"}[mode]


def _write_results(work, status, verdict):
    (work / "V3_RESULTS.json").write_text(json.dumps(
        {"verdict": verdict, "mode": status["mode"], "experiment_id": status["experiment_id"],
         "stages": status["stages"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# v3 real-domain experiment", "", f"Verdict: **{verdict}**",
             f"Mode: {status['mode']} · experiment: {status['experiment_id']}", "", "## Stages", ""]
    for s in status["stages"]:
        mark = {"ok": "✅", "planned": "•"}.get(s["status"], "⏭️" if "skipped" in s["status"] else "❌")
        lines.append(f"- {mark} {s['name']} ({s['status']}){' [gate]' if s.get('gate') else ''}")
    (work / "V3_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiments" / "v3_real_domain_generalization.json"))
    ap.add_argument("--manifest", default=str(ROOT / "configs" / "data_sources_v3.json"))
    ap.add_argument("--work-dir", default=str(ROOT / "outputs" / "v3-real-domain"))
    ap.add_argument("--mode", choices=["dry-run", "smoke", "full"], default="dry-run")
    ap.add_argument("--target-count", type=int, default=100000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run-id-prefix", default="v3")
    ap.add_argument("--train-causal", action="store_true")
    ap.add_argument("--train-bi-mntp", action="store_true")
    ap.add_argument("--train-reranker", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--allow-ml-smoke", action="store_true")
    ap.add_argument("--allow-research-failures", action="store_true",
                    help="let gate stages fail without aborting; marks verdict invalid_for_promotion")
    ap.add_argument("--i-understand-this-runs-gpu", action="store_true")
    args = ap.parse_args()

    # Stage 1 (fail fast): validate the v3 config + source manifest.
    cfg = load_v3_experiment_config(args.config)         # raises on invalid
    if not cfg.public_benchmarks_eval_only:
        print("ERROR: public_benchmarks_eval_only must be true.", file=sys.stderr)
        return 2
    dsa.load_v3_manifest(args.manifest)                  # raises on invalid (fail-closed)

    if args.mode == "full" and not args.i_understand_this_runs_gpu:
        print("ERROR: full mode requires --i-understand-this-runs-gpu", file=sys.stderr)
        return 2

    w = args.work_dir
    work = pathlib.Path(w)
    work.mkdir(parents=True, exist_ok=True)
    (work / "leakage").mkdir(exist_ok=True)
    stages = _stages(args, w)

    commands = ["# v3 real-domain experiment — planned commands", "",
                f"mode={args.mode} target_count={args.target_count} work_dir={w}", ""]
    for s in stages:
        tags = (" (GPU)" if s["gpu"] else "") + (" [GATE]" if s["gate"] else "")
        commands.append(f"## {s['name']}{tags}")
        commands.append("```\n" + " ".join(s["argv"]) + "\n```")
    (work / "COMMANDS.md").write_text("\n".join(commands) + "\n", encoding="utf-8")

    status = {"mode": args.mode, "experiment_id": cfg.experiment_id,
              "stages": [{"name": s["name"], "gpu": s["gpu"], "gate": s["gate"],
                          "status": "planned"} for s in stages]}

    print(f"[v3] {len(stages)} stages (mode={args.mode}); commands -> {work / 'COMMANDS.md'}")
    for s in stages:
        print(f"  - {s['name']}{' [GPU]' if s['gpu'] else ''}{' [GATE]' if s['gate'] else ''}")

    if args.mode == "dry-run":
        assert "torch" not in sys.modules, "dry-run must not import torch"
        (work / "STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        _write_results(work, status, "planned")
        print("dry-run-ok (no ML imports; nothing executed)")
        return 0

    run_gpu = (args.mode == "full") or args.allow_ml_smoke
    failed = invalid_for_promotion = False
    for s, entry in zip(stages, status["stages"]):
        if s["gpu"] and not run_gpu:
            entry["status"] = "skipped (gpu; printed only)"
            print(f"[skip] {s['name']}")
            continue
        if failed:
            entry["status"] = "skipped (prior failure)"
            continue
        print(f"[run] {s['name']}")
        rc = subprocess.run(s["argv"], cwd=str(ROOT)).returncode
        if rc == 0:
            entry["status"] = "ok"
        elif s["gate"] and args.allow_research_failures:
            entry["status"] = f"gate-failed(rc={rc}) — allowed (invalid_for_promotion)"
            invalid_for_promotion = True
            print(f"WARNING: gate {s['name']} failed but --allow-research-failures set; "
                  "verdict -> invalid_for_promotion", file=sys.stderr)
        else:
            entry["status"] = f"failed(rc={rc})"
            failed = True
            print(f"ERROR: stage {s['name']} failed (rc={rc}); stopping.", file=sys.stderr)

    verdict = _verdict(status["stages"], args.mode, invalid_for_promotion)
    status["verdict"] = verdict
    (work / "STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    _write_results(work, status, verdict)
    print(f"[v3] verdict={verdict}; STATUS -> {work / 'STATUS.json'}")
    return 1 if verdict == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
