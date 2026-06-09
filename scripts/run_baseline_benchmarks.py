#!/usr/bin/env python3
"""Reproducible benchmark runner for baselines, teachers, and Boldt student checkpoints (Prompt 9).

Models come from a config (`configs/baseline_models.json`), never a hard-coded list. Two eval
modes: ``local`` (local JSONL retrieval fixtures) and ``mteb`` (MTEB/MMTEB via the `mteb`
package when installed). Writes a JSON summary + a Markdown table, each row carrying full run
metadata.

`--dry-run` lists the planned model×task matrix and collects environment metadata WITHOUT
importing torch / downloading models. The `local_hashing` backend is a deterministic stdlib
stand-in for plumbing/tests — NOT a quality claim.
"""
from __future__ import annotations

import argparse
import importlib.metadata as ilm
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data_pipeline as dp  # noqa: E402
from boldt_embed.config_teacher import load_baseline_models_config  # noqa: E402
from boldt_embed.eval_harness import HashingEncoder, cosine_rank  # noqa: E402
from boldt_embed.metrics import aggregate, metrics_for_query  # noqa: E402

KS = (10, 100)


def _pkg_version(pkg):
    try:
        return ilm.version(pkg)  # reads metadata; does NOT import the package
    except Exception:
        return None


def collect_env_metadata():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    return {"commit": commit, "python": sys.version.split()[0],
            "platform": __import__("platform").platform(),
            "torch": _pkg_version("torch"), "transformers": _pkg_version("transformers"),
            "sentence_transformers": _pkg_version("sentence-transformers"),
            "mteb": _pkg_version("mteb")}


def _load_local(corpus_p, queries_p, qrels_p, limit):
    corpus = [{"id": str(r.get("doc_id") or r.get("id")),
               "text": r.get("document") or r.get("text") or ""} for r in dp.stream_jsonl(corpus_p)]
    qrels = {}
    for r in dp.stream_jsonl(qrels_p):
        if float(r.get("relevance", r.get("score", 1))) > 0:
            qrels.setdefault(str(r["query_id"]), set()).add(str(r.get("doc_id") or r.get("corpus-id")))
    queries = []
    for r in dp.stream_jsonl(queries_p):
        qid = str(r.get("query_id") or r.get("id"))
        if qid in qrels:
            queries.append({"query_id": qid, "query": r.get("query") or r.get("text") or "",
                            "positive_ids": qrels[qid]})
    if limit:
        queries = queries[:limit]
    return corpus, queries


def _encode(model_cfg, texts, device):
    if model_cfg.backend == "local_hashing":
        return HashingEncoder(dim=model_cfg.expected_dim or 256).encode(texts)
    if model_cfg.backend in ("sentence_transformers", "local_boldt"):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_cfg.model_name_or_path, device=device)
        return model.encode(texts, batch_size=model_cfg.batch_size,
                            normalize_embeddings=model_cfg.normalize,
                            show_progress_bar=False).tolist()
    raise NotImplementedError(
        f"backend '{model_cfg.backend}' not implemented; add an adapter or use "
        "sentence_transformers/local_boldt/local_hashing.")


def _eval_local(model_cfg, corpus, queries, device):
    q_texts = [(model_cfg.query_instruction or "") + q["query"] for q in queries]
    d_prefix = model_cfg.document_instruction or ""
    d_texts = [d_prefix + c["text"] for c in corpus]
    c_vecs = list(zip([c["id"] for c in corpus], _encode(model_cfg, d_texts, device)))
    q_vecs = _encode(model_cfg, q_texts, device)
    rows = []
    for i, q in enumerate(queries):
        ranked = cosine_rank(q_vecs[i], c_vecs)
        rows.append(metrics_for_query(ranked, set(q["positive_ids"]), KS))
    return aggregate(rows)


def _render_md(results, meta):
    lines = ["# Baseline benchmark report", "",
             f"commit: `{meta['commit']}` · torch: {meta['torch']} · "
             f"sentence-transformers: {meta['sentence_transformers']}", "",
             "| model | task | nDCG@10 | MRR@10 | Recall@10 | Recall@100 | MAP@10 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in results:
        m = r["metrics"]
        lines.append(f"| {r['model']} | {r['task']} | {m.get('ndcg@10')} | {m.get('mrr@10')} "
                     f"| {m.get('recall@10')} | {m.get('recall@100')} | {m.get('map@10')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=str(ROOT / "configs" / "baseline_models.json"))
    ap.add_argument("--tasks", default=str(ROOT / "benchmarks" / "mteb_german_tasks.json"))
    ap.add_argument("--mode", choices=["local", "mteb"], default="local")
    ap.add_argument("--task-name", default="local_retrieval")
    ap.add_argument("--eval-corpus", default=None)
    ap.add_argument("--eval-queries", default=None)
    ap.add_argument("--qrels", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these model ids/substrings")
    ap.add_argument("--output", default=str(ROOT / "outputs" / "baselines" / "baseline_report.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    models = load_baseline_models_config(args.models)
    if args.only:
        models = [m for m in models if any(s in m.model_name_or_path for s in args.only)]
    meta = collect_env_metadata()
    print(f"[env] {json.dumps(meta, ensure_ascii=False)}")
    print(f"[plan] {len(models)} models x mode={args.mode} task={args.task_name}")

    if args.dry_run:
        for m in models:
            print(f"  - {m.model_name_or_path} [{m.backend}] dim={m.expected_dim}")
        assert "torch" not in sys.modules, "dry-run must not import torch"
        print("dry-run-ok (no ML imports)")
        return 0

    if args.mode == "mteb":
        try:
            import mteb  # noqa: F401
        except ImportError as exc:
            raise SystemExit(f"mode=mteb needs the mteb package: pip install -e '.[eval]'. ({exc})")
        raise SystemExit("mteb mode: configure tasks in --tasks and run per the mteb API "
                         "(template in run_mteb_benchmark_template.py); not auto-run here.")

    if not (args.eval_corpus and args.eval_queries and args.qrels):
        print("ERROR: local mode needs --eval-corpus --eval-queries --qrels", file=sys.stderr)
        return 2
    corpus, queries = _load_local(args.eval_corpus, args.eval_queries, args.qrels, args.limit)
    print(f"[local] corpus={len(corpus)} queries={len(queries)}")

    results = []
    for m in models:
        try:
            metrics = _eval_local(m, corpus, queries, args.device)
        except Exception as exc:  # noqa: BLE001 - record per-model failure, keep going
            results.append({"model": m.model_name_or_path, "task": args.task_name,
                            "error": str(exc), "metrics": {}})
            print(f"  ! {m.model_name_or_path}: {exc}")
            continue
        results.append({"model": m.model_name_or_path, "task": args.task_name,
                        "backend": m.backend, "split": "test", "metrics": metrics})
        print(f"  {m.model_name_or_path:45s} ndcg@10={metrics.get('ndcg@10')}")

    report = {"status": "ok", "mode": args.mode, "task": args.task_name,
              "n_queries": len(queries), "n_corpus": len(corpus),
              "run_metadata": {"command": "scripts/run_baseline_benchmarks.py", **meta},
              "results": results}
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(_render_md(results, meta), encoding="utf-8")
    print(f"saved: {out} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
