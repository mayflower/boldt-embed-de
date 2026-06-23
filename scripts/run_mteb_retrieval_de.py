#!/usr/bin/env python3
"""Run MTEB(deu) RETRIEVAL tasks on a trained Boldt embedder (or any SentenceTransformers model),
restricted to the German subset.

Real benchmark — needs the eval extra (`pip install -e '.[eval]'`), dataset downloads and a GPU; it
is intentionally NOT part of the stdlib smoke gates. Per ADR-005 every number is written with run
metadata so it is auditable. The Boldt dense model uses NO query/doc prefix (symmetric), matching
the in-repo proxy eval, so no prompt is applied.

Retrieval-core default = GermanQuAD, GerDaLIR-Small, MIRACL (hard-negatives variant — the standard
affordable corpus), MultiLongDocRetrieval. Use --tasks to override (e.g. full MIRACLRetrieval).
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

DEFAULT_TASKS = [
    "GermanQuAD-Retrieval",
    "GerDaLIRSmall",
    "MIRACLRetrievalHardNegatives",
    "MultiLongDocRetrieval",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _scores_from_result(result) -> dict:
    """Best-effort extraction of the primary score per task from an mteb ModelResult."""
    out: dict = {}
    try:
        for tr in result.task_results:
            name = getattr(tr, "task_name", None) or getattr(getattr(tr, "task", None), "metadata", None)
            try:
                out[str(name)] = float(tr.get_score())
            except Exception:
                out[str(name)] = getattr(tr, "scores", None)
    except Exception as exc:  # fall back to the canonical cache files written by mteb
        out["_extract_error"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="SentenceTransformers path or HF id")
    ap.add_argument("--label", required=True, help="short name → outputs/mteb/<label>/")
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--langs", default="deu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-seq-length", type=int, default=256,
                    help="cap encode length; default 256 = the Boldt training length (also bounds "
                         "GPU memory on long-doc tasks like MLDR, which would OOM at batch 64×2048)")
    ap.add_argument("--loader", choices=["st", "mteb"], default="st",
                    help="'st' = bare SentenceTransformer (our prefix-free model); 'mteb' = "
                         "mteb.get_model, which applies the model's OFFICIAL prompts/pooling "
                         "(use for registry competitors e5/gte/Qwen so their numbers are fair)")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--config-kwargs", default=None,
                    help="st-loader only: JSON dict passed as SentenceTransformer config_kwargs "
                         "(e.g. disable a custom model's mem-efficient attention that asserts on CUDA)")
    ap.add_argument("--query-prompt", default=None,
                    help="st-loader only: prefix for queries (e.g. 'query:')")
    ap.add_argument("--doc-prompt", default=None,
                    help="st-loader only: prefix for documents (e.g. 'document:'); mapped to both "
                         "the 'passage' and 'document' prompt names mteb may request")
    ap.add_argument("--output-dir", default="outputs/mteb")
    args = ap.parse_args()

    import mteb
    from sentence_transformers import SentenceTransformer

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    out = Path(args.output_dir) / args.label
    out.mkdir(parents=True, exist_ok=True)

    if args.loader == "mteb":
        # mteb's registry wrapper applies each model's official prompts/pooling/instruction.
        model = mteb.get_model(args.model)
        base = getattr(model, "model", None)
        if base is not None:  # cap the underlying encoder length for feasibility/fairness
            try:
                base.max_seq_length = args.max_seq_length
            except Exception:
                pass
    else:
        st_kwargs = {"trust_remote_code": True} if args.trust_remote_code else {}
        if args.config_kwargs:
            st_kwargs["config_kwargs"] = json.loads(args.config_kwargs)
        model = SentenceTransformer(args.model, **st_kwargs)
        model.max_seq_length = args.max_seq_length
        if args.query_prompt is not None or args.doc_prompt is not None:
            qp, dp = args.query_prompt or "", args.doc_prompt or ""
            # cover every prompt_name mteb may pass for the corpus side (passage/document)
            model.prompts = {"query": qp, "passage": dp, "document": dp}
    tasks = mteb.get_tasks(tasks=task_names)
    filtered = []
    for t in tasks:  # restrict multilingual tasks to German (no-op for German-only tasks)
        try:
            t = t.filter_languages(langs)
        except Exception:
            pass
        filtered.append(t)

    meta = {
        "command": "run_mteb_retrieval_de.py",
        "commit": _git_commit(),
        "model": args.model,
        "label": args.label,
        "tasks": task_names,
        "langs": langs,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "loader": args.loader,
        "query_prompt": args.query_prompt,
        "doc_prompt": args.doc_prompt,
        "hardware": platform.platform(),
        "mteb_version": mteb.__version__,
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    started = time.time()
    cache = mteb.ResultCache(cache_path=str(out / "cache"))
    result = mteb.evaluate(model, filtered, cache=cache,
                           encode_kwargs={"batch_size": args.batch_size})
    elapsed = round(time.time() - started, 1)

    scores = _scores_from_result(result)
    summary = {"meta": meta, "elapsed_seconds": elapsed, "scores": scores}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    print(json.dumps({"label": args.label, "elapsed_seconds": elapsed, "scores": scores},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
