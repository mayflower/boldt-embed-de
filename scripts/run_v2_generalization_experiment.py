#!/usr/bin/env python3
"""Reproducible v2 data-scale-generalization orchestrator. Defaults to SAFE dry-run; only `full`
mode (with --i-understand-this-runs-gpu) executes real GPU work. It orchestrates the existing
scripts — it does not duplicate their logic. Writes COMMANDS.md + STATUS.json under --work-dir.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import source_manifest as sm  # noqa: E402
from boldt_embed.v2_experiment_config import load_v2_experiment_config  # noqa: E402

PY = sys.executable
S = str(ROOT / "scripts")


def _stages(args, w):
    """Ordered stages: {name, argv, gpu}. argv are lists for subprocess (no shell)."""
    tc = f"{w}/teacher-cache/qwen3_v2"
    cand = f"{w}/candidates_v2.jsonl"
    filt = f"{tc}.filtered.jsonl"
    st = []
    st.append({"name": "build_candidates", "gpu": False, "argv": [
        PY, f"{S}/build_v2_candidates.py", "--manifest", args.manifest,
        "--source-jsonl", "data/raw/v2/*.jsonl", "--output", cand,
        "--domain-config", args.config, "--target-count", str(args.target_count),
        "--dedup", "--pii-scan"]})
    st.append({"name": "generate_synthetic", "gpu": False, "argv": [
        PY, f"{S}/generate_synthetic_queries.py", "--passages", f"{w}/passages.jsonl",
        "--output", f"{w}/synthetic_v2.jsonl", "--families", "germanquad", "web", "faq", "admin",
        "--queries-per-passage", "4"]})
    st.append({"name": "teacher_cache", "gpu": True, "argv": [
        PY, f"{S}/build_teacher_cache.py", "--input", cand, "--output", f"{tc}.jsonl",
        "--mode", args.teacher_mode, "--shard-size", "5000", "--max-length", "512"]})
    st.append({"name": "summarize_filter_cache", "gpu": False, "argv": [
        PY, f"{S}/summarize_teacher_cache.py", "--input", f"{tc}.manifest.json",
        "--output", f"{tc}.summary.json", "--filter-output", filt, "--reranker-threshold", "2.0"]})
    st.append({"name": "mine_hard_negatives", "gpu": False, "argv": [
        PY, f"{S}/mine_hard_negatives_2026.py", "--candidates", cand, "--teacher-cache", filt,
        "--output", f"{w}/hardneg_v2.jsonl", "--negatives-per-query", "8"]})
    st.append({"name": "reranker_candidate_lists", "gpu": False, "argv": [
        PY, f"{S}/build_reranker_candidates_v2.py", "--candidates", cand, "--teacher-cache", filt,
        "--output", f"{w}/reranker_train_v2.jsonl"]})
    if args.train_causal:
        st.append({"name": "train_causal", "gpu": True, "argv": [
            PY, f"{S}/train_modern_embedder.py", "--teacher-cache", filt,
            "--hard-negatives", f"{w}/hardneg_v2.jsonl",
            "--output", f"{w}/checkpoints/boldt-modern-causal-v2", "--bf16",
            "--gradient-checkpointing", "--run-id", f"{args.run_id_prefix}-causal"]})
    if args.train_bi_mntp:
        st.append({"name": "prepare_bi_mntp", "gpu": True, "argv": [
            PY, f"{S}/prepare_bidirectional_student.py", "--texts", f"{w}/mntp_texts.jsonl",
            "--output", f"{w}/checkpoints/boldt-bi-mntp-v2", "--bf16", "--gradient-checkpointing",
            "--run-id", f"{args.run_id_prefix}-mntp"]})
        st.append({"name": "train_bi_mntp", "gpu": True, "argv": [
            PY, f"{S}/train_modern_embedder.py", "--base-model", f"{w}/checkpoints/boldt-bi-mntp-v2",
            "--bidirectional", "true", "--teacher-cache", filt,
            "--output", f"{w}/checkpoints/boldt-modern-bi-mntp-v2", "--bf16",
            "--gradient-checkpointing", "--run-id", f"{args.run_id_prefix}-bi-mntp"]})
    if args.train_reranker:
        st.append({"name": "train_reranker", "gpu": True, "argv": [
            PY, f"{S}/train_modern_reranker.py", "--candidate-lists", f"{w}/reranker_train_v2.jsonl",
            "--loss", "mixed", "--output", f"{w}/checkpoints/boldt-reranker-modern-v2", "--bf16",
            "--run-id", f"{args.run_id_prefix}-reranker"]})
    if args.eval:
        for ds in ("germanquad", "dt_test", "gerdalir"):
            st.append({"name": f"eval_dense_{ds}", "gpu": True, "argv": [
                PY, f"{S}/run_baseline_benchmarks.py", "--models", "data/processed/v2_eval_models.json",
                "--mode", "local", "--task-name", ds,
                "--eval-corpus", f"data/processed/eval/{ds}_corpus.jsonl",
                "--eval-queries", f"data/processed/eval/{ds}_queries.jsonl",
                "--qrels", f"data/processed/eval/{ds}_qrels.jsonl",
                "--output", f"{w}/eval/dense_{ds}.json", "--run-id", f"{args.run_id_prefix}-eval-{ds}"]})
        st.append({"name": "summarize_results", "gpu": False, "argv": [
            PY, f"{S}/summarize_v2_results.py", "--v1-dir", "outputs", "--v2-dir", w,
            "--config", args.config, "--output", f"{w}/V2_RESULTS.md",
            "--json-output", f"{w}/V2_RESULTS.json"]})
    return st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiments" / "v2_generalization.json"))
    ap.add_argument("--manifest", default=str(ROOT / "configs" / "data_sources_v2.json"))
    ap.add_argument("--work-dir", default=str(ROOT / "outputs" / "v2-generalization"))
    ap.add_argument("--mode", choices=["dry-run", "smoke", "full"], default="dry-run")
    ap.add_argument("--target-count", type=int, default=50000)
    ap.add_argument("--teacher-mode", choices=["embedding", "reranker", "both"], default="both")
    ap.add_argument("--train-causal", action="store_true")
    ap.add_argument("--train-bi-mntp", action="store_true")
    ap.add_argument("--train-reranker", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run-id-prefix", default="v2")
    ap.add_argument("--allow-ml-smoke", action="store_true")
    ap.add_argument("--i-understand-this-runs-gpu", action="store_true")
    args = ap.parse_args()

    # Fail fast: invalid config (eval-only enforced inside the loader) or manifest.
    cfg = load_v2_experiment_config(args.config)
    if not cfg.public_benchmarks_eval_only:
        print("ERROR: public_benchmarks_eval_only must be true.", file=sys.stderr)
        return 2
    sm.load_source_manifest(args.manifest)

    w = args.work_dir
    stages = _stages(args, w)
    work = pathlib.Path(w)
    work.mkdir(parents=True, exist_ok=True)
    commands_md = ["# v2 experiment — planned commands", "",
                   f"mode={args.mode} target_count={args.target_count} work_dir={w}", ""]
    for s in stages:
        commands_md.append(f"## {s['name']}{' (GPU)' if s['gpu'] else ''}")
        commands_md.append("```\n" + " ".join(s["argv"]) + "\n```")
    (work / "COMMANDS.md").write_text("\n".join(commands_md) + "\n", encoding="utf-8")
    status = {"mode": args.mode, "experiment_id": cfg.experiment_id,
              "stages": [{"name": s["name"], "gpu": s["gpu"], "status": "planned"} for s in stages]}

    print(f"[v2] {len(stages)} stages (mode={args.mode}); commands -> {work / 'COMMANDS.md'}")
    for s in stages:
        print(f"  - {s['name']}{' [GPU]' if s['gpu'] else ''}")

    if args.mode == "dry-run":
        (work / "STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        assert "torch" not in sys.modules, "dry-run must not import torch"
        print("dry-run-ok (no ML imports; nothing executed)")
        return 0

    if args.mode == "full" and not args.i_understand_this_runs_gpu:
        print("ERROR: full mode requires --i-understand-this-runs-gpu", file=sys.stderr)
        return 2

    run_gpu = (args.mode == "full") or args.allow_ml_smoke
    failed = False
    for s, entry in zip(stages, status["stages"]):
        if s["gpu"] and not run_gpu:
            entry["status"] = "skipped (gpu; printed only)"
            print(f"[skip] {s['name']}: {' '.join(s['argv'])}")
            continue
        if failed:
            entry["status"] = "skipped (prior failure)"
            continue
        print(f"[run] {s['name']}")
        rc = subprocess.run(s["argv"], cwd=str(ROOT)).returncode
        entry["status"] = "ok" if rc == 0 else f"failed(rc={rc})"
        if rc != 0:
            failed = True
            print(f"ERROR: stage {s['name']} failed (rc={rc}); stopping.", file=sys.stderr)
    (work / "STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"[v2] STATUS -> {work / 'STATUS.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
