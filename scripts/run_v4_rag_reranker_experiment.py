#!/usr/bin/env python3
"""One-command v4 German RAG reranker experiment. SAFE by default: ``dry-run`` only plans (no
torch, no downloads); ``full`` (with --i-understand-this-runs-gpu) executes. Optimizes directly
for a RAG reranker — it does NOT require legal/admin corpora, and GerDaLIR is diagnostic-only.

Real first stage is BM25 (build/search_bm25_index). Dense (v3 causal) / e5 / qwen are optional
candidate-source diversity: pass prebuilt per-query result files via --dense-results etc.
(see the runbook); they're consumed by the candidate-list build, never fabricated here.

Writes COMMANDS.md + STATUS.json + V4_RAG_RESULTS.{md,json} under --work-dir.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed.v4_rag_config import load_v4_rag_config  # noqa: E402

PY = sys.executable
S = str(ROOT / "scripts")
PUB_EVAL = str(ROOT / "data" / "processed" / "eval")          # germanquad/dt_test/gerdalir fixtures
LOCAL_RAG = ROOT / "data" / "eval" / "rag_local"


def _eval_sets(args):
    """[(name, corpus, queries, qrels, diagnostic)] — webfaq + public benchmarks + optional."""
    w = args.work_dir
    sets = [("webfaq", f"{w}/eval/webfaq/corpus.jsonl", f"{w}/eval/webfaq/queries.jsonl",
             f"{w}/eval/webfaq/qrels.jsonl", False)]
    for name in ("germanquad", "dt_test"):
        sets.append((name, f"{PUB_EVAL}/{name}_corpus.jsonl", f"{PUB_EVAL}/{name}_queries.jsonl",
                     f"{PUB_EVAL}/{name}_qrels.jsonl", False))
    if LOCAL_RAG.exists():
        sets.append(("local_rag", str(LOCAL_RAG / "corpus.jsonl"), str(LOCAL_RAG / "queries.jsonl"),
                     str(LOCAL_RAG / "qrels.jsonl"), False))
    if args.with_gerdalir_diagnostic:
        sets.append(("gerdalir", f"{PUB_EVAL}/gerdalir_corpus.jsonl",
                     f"{PUB_EVAL}/gerdalir_queries.jsonl", f"{PUB_EVAL}/gerdalir_qrels.jsonl", True))
    return sets


def _stages(args):
    w = args.work_dir
    fs = f"{w}/first_stage"
    cl = f"{w}/candidate_lists"
    tc = f"{w}/teacher"
    ckpt = f"{w}/checkpoints/boldt-rag-reranker-v4"
    eval_dir = f"{w}/eval"
    webfaq_train = f"{w}/train/webfaq"
    st = []

    def _opt_sources(set_name):
        """optional pre-built dense/e5/qwen result files for this set (consumed if provided)."""
        argv = []
        for flag, name in ((args.dense_results, "--dense-results"), (args.e5_results, "--e5-results"),
                           (args.qwen_results, "--qwen-results")):
            if flag:
                argv += [name, flag.format(set=set_name)]
        return argv

    # 2. WebFAQ held-out eval split (test) + 3. WebFAQ train split
    st.append({"name": "build_webfaq_eval", "gpu": False, "gate": False, "argv": [
        PY, f"{S}/build_rag_eval_sets.py", "--mode", "webfaq", "--faq-input", args.faq_input,
        "--split", "test", "--output-dir", f"{eval_dir}/webfaq"]})
    st.append({"name": "build_webfaq_train", "gpu": False, "gate": False, "argv": [
        PY, f"{S}/build_rag_eval_sets.py", "--mode", "webfaq", "--faq-input", args.faq_input,
        "--split", "train", "--output-dir", webfaq_train]})
    if LOCAL_RAG.exists():
        st.append({"name": "build_local_rag_eval", "gpu": False, "gate": False, "argv": [
            PY, f"{S}/build_rag_eval_sets.py", "--mode", "local", "--output-dir", f"{eval_dir}/local_rag"]})

    # 4-7. BM25 first stage + fixed candidate lists, for the TRAIN set and every EVAL set
    st.append({"name": "bm25_index_train", "gpu": False, "gate": False, "argv": [
        PY, f"{S}/build_bm25_index.py", "--corpus", f"{webfaq_train}/corpus.jsonl",
        "--output", f"{fs}/bm25_train.json"]})
    st.append({"name": "bm25_search_train", "gpu": False, "gate": False, "argv": [
        PY, f"{S}/search_bm25_index.py", "--index", f"{fs}/bm25_train.json",
        "--queries", f"{webfaq_train}/queries.jsonl", "--top-k", "20",
        "--output", f"{fs}/bm25_train.results.jsonl"]})
    st.append({"name": "build_train_candidate_lists", "gpu": False, "gate": False, "argv": [
        PY, f"{S}/build_rag_candidate_lists.py", "--queries", f"{webfaq_train}/queries.jsonl",
        "--corpus", f"{webfaq_train}/corpus.jsonl", "--qrels", f"{webfaq_train}/qrels.jsonl",
        "--bm25-results", f"{fs}/bm25_train.results.jsonl", *_opt_sources("train"),
        "--mode", "train", "--output", f"{cl}/rag_reranker_train_lists.jsonl"]})

    # 8. teacher-score the train lists; 9. train the v4 reranker (with eval-leakage guard)
    st.append({"name": "teacher_score_train", "gpu": True, "gate": False, "argv": [
        PY, f"{S}/score_rag_candidate_lists.py", "--input", f"{cl}/rag_reranker_train_lists.jsonl",
        "--mode", "reranker", "--output", f"{tc}/rag_train_scored.jsonl",
        "--summary", f"{tc}/rag_teacher_summary.json"]})
    st.append({"name": "train_rag_reranker", "gpu": True, "gate": False, "argv": [
        PY, f"{S}/train_rag_reranker_v4.py", "--candidate-lists", f"{tc}/rag_train_scored.jsonl",
        "--loss", "mixed_listwise", "--bf16", "--gradient-checkpointing", "--epochs", "1",
        "--batch-size", "8", "--eval-query-ids", f"{eval_dir}/webfaq/queries.jsonl",
        "--output", ckpt, "--run-id", f"{args.run_id_prefix}-rag-reranker"]})

    # 10. per-eval-set fixed candidate lists + reranker lift
    for name, corpus, queries, qrels, diag in _eval_sets(args):
        if name not in ("webfaq", "local_rag"):     # webfaq/local built above; public sets here
            pass
        st.append({"name": f"bm25_index_{name}", "gpu": False, "gate": False, "argv": [
            PY, f"{S}/build_bm25_index.py", "--corpus", corpus, "--output", f"{fs}/bm25_{name}.json"]})
        st.append({"name": f"bm25_search_{name}", "gpu": False, "gate": False, "argv": [
            PY, f"{S}/search_bm25_index.py", "--index", f"{fs}/bm25_{name}.json",
            "--queries", queries, "--top-k", "20", "--output", f"{fs}/bm25_{name}.results.jsonl"]})
        st.append({"name": f"candidate_lists_{name}", "gpu": False, "gate": False, "argv": [
            PY, f"{S}/build_rag_candidate_lists.py", "--queries", queries, "--corpus", corpus,
            "--qrels", qrels, "--bm25-results", f"{fs}/bm25_{name}.results.jsonl",
            *_opt_sources(name), "--mode", "eval", "--output", f"{cl}/eval_{name}_lists.jsonl"]})
        lift = {"name": f"lift_{name}", "gpu": True, "gate": False, "argv": [
            PY, f"{S}/eval_rag_reranker_lift.py", "--reranker", ckpt,
            "--candidate-lists", f"{cl}/eval_{name}_lists.jsonl", "--eval-set", name,
            "--output", f"{eval_dir}/reranker_lift_{name}.json"]}
        if diag:
            lift["argv"].append("--diagnostic")      # GerDaLIR: reported, never gates
        st.append(lift)

    # 11. promotion gate (GerDaLIR ignored); 12. summary written by the orchestrator
    st.append({"name": "promotion_gate", "gpu": False, "gate": True, "argv": [
        PY, f"{S}/check_rag_reranker_promotion_gate.py", "--eval-dir", eval_dir,
        "--output", f"{eval_dir}/rag_reranker_gate.json",
        "--markdown", f"{eval_dir}/rag_reranker_gate.md"]})
    return st


def _verdict(stages, mode, invalid):
    if any(s["status"].startswith("failed") for s in stages):
        return "failed"
    if invalid:
        return "invalid_for_promotion"
    return {"dry-run": "planned", "smoke": "smoke-ok", "full": "promotable"}[mode]


def _write_results(work, status, verdict):
    (work / "V4_RAG_RESULTS.json").write_text(json.dumps(
        {"verdict": verdict, "mode": status["mode"], "experiment_id": status["experiment_id"],
         "stages": status["stages"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# v4 German RAG reranker experiment", "", f"Verdict: **{verdict}**",
             f"Mode: {status['mode']} · experiment: {status['experiment_id']}", "",
             "Optimizes a RAG reranker (WebFAQ/local-RAG lift; GermanQuAD/DT-test neutral-or-better);"
             " legal/admin are NOT required and GerDaLIR is diagnostic-only.", "", "## Stages", ""]
    for s in status["stages"]:
        mark = {"ok": "✅", "planned": "•"}.get(s["status"], "⏭️" if "skipped" in s["status"] else "❌")
        lines.append(f"- {mark} {s['name']} ({s['status']}){' [gate]' if s.get('gate') else ''}")
    (work / "V4_RAG_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiments" / "v4_rag_reranker.json"))
    ap.add_argument("--work-dir", default=str(ROOT / "outputs" / "v4-rag-reranker"))
    ap.add_argument("--faq-input", default=str(ROOT / "data" / "raw" / "v3" / "faq_real_local.jsonl"),
                    help="real WebFAQ FAQ rows (query/answer) to split into train/held-out")
    ap.add_argument("--mode", choices=["dry-run", "smoke", "full"], default="dry-run")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run-id-prefix", default="v4")
    ap.add_argument("--dense-results", default=None, help="prebuilt v3-dense result file pattern, e.g. .../dense_{set}.jsonl")
    ap.add_argument("--e5-results", default=None)
    ap.add_argument("--qwen-results", default=None)
    ap.add_argument("--with-gerdalir-diagnostic", action="store_true",
                    help="also run a GerDaLIR lift (DIAGNOSTIC only — never gates)")
    ap.add_argument("--allow-ml-smoke", action="store_true")
    ap.add_argument("--allow-research-failures", action="store_true",
                    help="let the gate fail without aborting; verdict -> invalid_for_promotion")
    ap.add_argument("--i-understand-this-runs-gpu", action="store_true")
    args = ap.parse_args()

    # Stage 1 (fail fast): validate the v4 config (legal diagnostic-only; eval-only public bench).
    cfg = load_v4_rag_config(args.config)
    if not cfg.legal_eval_is_diagnostic_only:
        print("ERROR: legal_eval_is_diagnostic_only must be true.", file=sys.stderr)
        return 2
    if args.mode == "full" and not args.i_understand_this_runs_gpu:
        print("ERROR: full mode requires --i-understand-this-runs-gpu", file=sys.stderr)
        return 2

    work = pathlib.Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    stages = _stages(args)

    commands = ["# v4 RAG reranker experiment — planned commands", "",
                f"mode={args.mode} work_dir={args.work_dir} config={cfg.experiment_id}", ""]
    for s in stages:
        tags = (" (GPU)" if s["gpu"] else "") + (" [GATE]" if s["gate"] else "")
        commands.append(f"## {s['name']}{tags}")
        commands.append("```\n" + " ".join(s["argv"]) + "\n```")
    (work / "COMMANDS.md").write_text("\n".join(commands) + "\n", encoding="utf-8")
    status = {"mode": args.mode, "experiment_id": cfg.experiment_id,
              "stages": [{"name": s["name"], "gpu": s["gpu"], "gate": s["gate"],
                          "status": "planned"} for s in stages]}

    print(f"[v4-rag] {len(stages)} stages (mode={args.mode}); commands -> {work / 'COMMANDS.md'}")
    for s in stages:
        print(f"  - {s['name']}{' [GPU]' if s['gpu'] else ''}{' [GATE]' if s['gate'] else ''}")

    if args.mode == "dry-run":
        assert "torch" not in sys.modules, "dry-run must not import torch"
        (work / "STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        _write_results(work, status, "planned")
        print("dry-run-ok (no ML imports; nothing executed)")
        return 0

    run_gpu = (args.mode == "full") or args.allow_ml_smoke
    failed = invalid = False
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
            invalid = True
        else:
            entry["status"] = f"failed(rc={rc})"
            failed = True
            print(f"ERROR: stage {s['name']} failed (rc={rc}); stopping.", file=sys.stderr)
    verdict = _verdict(status["stages"], args.mode, invalid)
    status["verdict"] = verdict
    (work / "STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    _write_results(work, status, verdict)
    print(f"[v4-rag] verdict={verdict}; STATUS -> {work / 'STATUS.json'}")
    return 1 if verdict == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
