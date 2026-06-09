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
