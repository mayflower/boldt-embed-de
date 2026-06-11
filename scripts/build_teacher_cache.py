#!/usr/bin/env python3
"""Build a teacher-score cache for the 2026 distillation workflow.

Reads candidate (query, document) rows, scores them with the Qwen3 embedding and/or
reranker teacher(s) configured in ``configs/teacher_models.json``, and writes a JSONL cache.

* ``--dry-run`` validates the input schema and prints the first 3 *planned* rows WITHOUT
  importing torch / sentence_transformers (no model download, no GPU).
* ``--resume`` skips (query_id, doc_id) pairs already present in the output.

Real teacher inference (anything but ``--dry-run``) requires the ``train`` extras + a GPU.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import experiment_registry as ER  # noqa: E402
from boldt_embed import teacher as T  # noqa: E402  (stdlib-only at import time)
from boldt_embed.config_teacher import load_teacher_models_config  # noqa: E402


def _validate_input(candidates):
    problems = []
    for i, row in enumerate(candidates):
        for err in T.validate_candidate_record(row):
            problems.append(f"row {i}: {err}")
    return problems


def _run_sharded(args, cfg, candidates, emb_name, rr_name) -> int:
    """Sharded teacher scoring: <out_dir>/<prefix>.shard-NNNNN.jsonl + <prefix>.manifest.json.
    Resume skips already-scored rows per shard; --shard-index runs a single shard."""
    out_dir = pathlib.Path(args.output).parent
    prefix = pathlib.Path(args.output).name
    if prefix.endswith(".jsonl"):
        prefix = prefix[:-6]
    shards = T.shard_candidates(candidates, args.shard_size)
    indices = [args.shard_index] if args.shard_index is not None else list(range(len(shards)))
    planned = [(i, T.shard_path(out_dir, prefix, i), len(shards[i]))
               for i in indices if 0 <= i < len(shards)]
    print(f"[shard] {len(shards)} shard(s) of <= {args.shard_size}; running {len(planned)} "
          f"(mode={args.mode})")
    man_path = out_dir / f"{prefix}.manifest.json"

    if args.dry_run:
        for i, p, n in planned[:8]:
            print(f"  shard {i:05d}: {n} rows -> {p}")
        print(f"  manifest -> {man_path}")
        assert "torch" not in sys.modules, "dry-run must not import torch"
        assert "sentence_transformers" not in sys.modules, "dry-run must not import sentence_transformers"
        print("dry-run-ok (no ML imports)")
        return 0

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"Real scoring needs extras: pip install -e '.[train]'. ({exc})")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"prefix": prefix, "mode": args.mode, "shard_size": args.shard_size,
                "n_shards": len(shards), "shards": []}
    total = 0
    for i, p, _ in planned:
        shard = shards[i]
        if args.resume:
            shard = T.filter_unscored(shard, T.existing_cache_keys(p))
        if shard:
            rows = T.score_candidates_for_queries(
                shard, cfg, args.mode, device=args.device,
                batch_size_embedding=args.batch_size_embedding,
                batch_size_reranker=args.batch_size_reranker)
            w = T.write_teacher_cache_jsonl(p, rows, append=args.resume)
            total += w
            print(f"  shard {i:05d}: wrote {w} -> {p}")
        else:
            print(f"  shard {i:05d}: up to date")
        manifest["shards"].append({"index": i, "path": str(p), "rows": len(shards[i])})
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    card = ER.emit_run_card(args.run_id, "teacher_cache", "scripts/build_teacher_cache.py",
                            model=f"{emb_name} + {rr_name}", dataset=args.input,
                            metrics={"rows_written": total, "n_shards": len(planned)},
                            input_artifacts=[args.input], output_artifacts=[str(man_path)],
                            notes=f"sharded mode={args.mode} shard_size={args.shard_size}")
    print(f"[shard] wrote {total} rows across {len(planned)} shard(s); manifest -> {man_path}; "
          f"run card: {card}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-config", default=str(ROOT / "configs" / "teacher_models.json"))
    ap.add_argument("--input", default=str(ROOT / "data" / "processed" / "candidates.jsonl"))
    ap.add_argument("--output", default=str(ROOT / "outputs" / "teacher-cache" / "teacher_scores.jsonl"))
    ap.add_argument("--mode", choices=["embedding", "reranker", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch-size-embedding", type=int, default=None)
    ap.add_argument("--batch-size-reranker", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=None, help="override teacher max_length")
    ap.add_argument("--shard-size", type=int, default=0, help=">0 enables sharded output")
    ap.add_argument("--shard-index", type=int, default=None, help="score only this shard (parallel runs)")
    ap.add_argument("--run-id", default=None, help="experiment run id (run card written on success)")
    args = ap.parse_args()

    cfg = load_teacher_models_config(args.teacher_config)
    emb_name = cfg.embedding_teacher.model_name
    rr_name = cfg.reranker_teacher.model_name

    if not pathlib.Path(args.input).exists():
        print(f"ERROR: input candidates file not found: {args.input}", file=sys.stderr)
        return 2
    candidates = T.read_candidates(args.input, limit=args.limit)
    problems = _validate_input(candidates)
    print(f"[input] {len(candidates)} candidate rows; schema problems: {len(problems)}")
    for p in problems[:10]:
        print(f"  - {p}")
    if problems:
        print("ERROR: fix candidate schema problems before scoring.", file=sys.stderr)
        return 2

    if args.max_length:  # memory knob for v2 scale
        cfg.embedding_teacher.max_length = args.max_length
        cfg.reranker_teacher.max_length = args.max_length

    if args.shard_size and args.shard_size > 0:
        return _run_sharded(args, cfg, candidates, emb_name, rr_name)

    if args.resume:
        done = T.existing_cache_keys(args.output)
        before = len(candidates)
        candidates = T.filter_unscored(candidates, done)
        print(f"[resume] {len(done)} already cached; {before - len(candidates)} skipped, "
              f"{len(candidates)} to score")

    if args.dry_run:
        preview = T.plan_preview_rows(candidates, args.mode, emb_name, rr_name, n=3)
        print(f"=== DRY RUN: would score {len(candidates)} rows "
              f"(mode={args.mode}, embedding={emb_name}, reranker={rr_name}) ===")
        print("first 3 planned cache rows:")
        for row in preview:
            print(json.dumps(row, ensure_ascii=False))
        # Hard guarantee: dry-run must not have imported ML libraries.
        assert "torch" not in sys.modules, "dry-run must not import torch"
        assert "sentence_transformers" not in sys.modules, "dry-run must not import sentence_transformers"
        print("dry-run-ok (no ML imports)")
        return 0

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"Real scoring needs extras: pip install -e '.[train]'. ({exc})")

    if not candidates:
        print("[score] nothing to score; cache is up to date.")
        return 0
    print(f"=== Scoring {len(candidates)} rows (mode={args.mode}) on {args.device} ===")
    rows = T.score_candidates_for_queries(
        candidates, cfg, args.mode, device=args.device,
        batch_size_embedding=args.batch_size_embedding,
        batch_size_reranker=args.batch_size_reranker)
    n = T.write_teacher_cache_jsonl(args.output, rows, append=args.resume)
    card = ER.emit_run_card(args.run_id, "teacher_cache", "scripts/build_teacher_cache.py",
                            model=f"{emb_name} + {rr_name}", dataset=args.input,
                            metrics={"rows_written": n}, input_artifacts=[args.input],
                            output_artifacts=[args.output], notes=f"mode={args.mode}")
    print(f"[score] wrote {n} cache rows to {args.output} (append={args.resume}); run card: {card}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
