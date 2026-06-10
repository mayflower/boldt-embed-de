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
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data_pipeline as dp  # noqa: E402
from boldt_embed import experiment_registry as ER  # noqa: E402
from boldt_embed.config_teacher import load_baseline_models_config  # noqa: E402
from boldt_embed.eval_harness import HashingEncoder, cosine_rank  # noqa: E402
from boldt_embed.experiment_registry import collect_env_metadata  # noqa: E402  (canonical)
from boldt_embed.metrics import aggregate, metrics_for_query  # noqa: E402

KS = (10, 100)


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
    # Char-truncate BEFORE tokenization: some corpora (e.g. legal) have 100k+ char docs;
    # feeding those whole to the tokenizer is a CPU bottleneck even though max_seq_length
    # truncates afterward. ~8 chars/token * max_length gives a safe upper bound.
    cap = max(int(model_cfg.max_length) * 8, 2048)
    texts = [t[:cap] for t in texts]
    if model_cfg.backend == "local_hashing":
        return HashingEncoder(dim=model_cfg.expected_dim or 256).encode(texts)
    if model_cfg.backend in ("sentence_transformers", "local_boldt"):
        from sentence_transformers import SentenceTransformer
        bi = bool(model_cfg.raw.get("bidirectional"))
        st_kwargs = {"model_kwargs": {"attn_implementation": "eager"}} if bi else {}
        model = SentenceTransformer(model_cfg.model_name_or_path, device=device, **st_kwargs)
        if bi:
            # Re-apply the LLM2Vec bidirectional patch — runtime, not saved in weights — so a
            # bidirectional student is actually bidirectional at eval (matches training).
            from boldt_embed.train_modern import apply_bidirectional_to_st
            apply_bidirectional_to_st(model)
        # Cap sequence length — long legal docs at a model's native (2k+) max_seq_length
        # explode compute/memory; the configured max_length is the eval setting.
        try:
            model.max_seq_length = int(model_cfg.max_length)
        except Exception:
            pass
        return model.encode(texts, batch_size=model_cfg.batch_size,
                            normalize_embeddings=model_cfg.normalize,
                            show_progress_bar=False).tolist()
    raise NotImplementedError(
        f"backend '{model_cfg.backend}' not implemented; add an adapter or use "
        "sentence_transformers/local_boldt/local_hashing.")


def _eval_local(model_cfg, corpus, queries, device):
    import torch

    q_texts = [(model_cfg.query_instruction or "") + q["query"] for q in queries]
    d_prefix = model_cfg.document_instruction or ""
    d_texts = [d_prefix + c["text"] for c in corpus]
    cids = [c["id"] for c in corpus]
    # Embeddings are L2-normalized -> cosine = dot product. Rank on the GPU via matmul;
    # the previous pure-Python ranking was O(queries*corpus*dim) and unusable at corpus scale.
    dev = device if (device and device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    c_t = torch.tensor(_encode(model_cfg, d_texts, device), dtype=torch.float32, device=dev)
    q_t = torch.tensor(_encode(model_cfg, q_texts, device), dtype=torch.float32, device=dev)
    topk = min(200, len(cids))
    rows = []
    for start in range(0, q_t.size(0), 256):
        sims = q_t[start:start + 256] @ c_t.t()
        idx = torch.topk(sims, topk, dim=1).indices.tolist()
        for j, row in enumerate(idx):
            ranked = [cids[k] for k in row]
            rows.append(metrics_for_query(ranked, set(queries[start + j]["positive_ids"]), KS))
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
    ap.add_argument("--run-id", default=None, help="experiment run id (run card written on success)")
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
    best = next((r for r in results if r.get("metrics")), {})
    card = ER.emit_run_card(args.run_id, "eval", "scripts/run_baseline_benchmarks.py",
                            dataset=args.task_name, metrics=best.get("metrics"),
                            input_artifacts=[args.eval_corpus, args.eval_queries, args.qrels],
                            output_artifacts=[str(out)],
                            notes=f"baseline benchmark, {len(results)} models, mode={args.mode}")
    print(f"saved: {out} and {out.with_suffix('.md')}; run card: {card}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
