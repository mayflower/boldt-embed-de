#!/usr/bin/env python3
"""Train the Boldt student embedder with the modern SBERT loss stack (Prompt 4).

Input is the teacher cache (`outputs/teacher-cache/*.jsonl`). `--dry-run` validates the
student config, reads the first rows of the cache, builds dataset metadata, and prints the
planned loss stack WITHOUT importing torch / sentence_transformers. Real training needs the
`train` extras + a GPU.
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
from boldt_embed import train_modern as TM  # noqa: E402
from boldt_embed.config_teacher import load_student_training_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student-config", default=str(ROOT / "configs" / "student_training_2026.json"))
    ap.add_argument("--teacher-cache", default=str(ROOT / "outputs" / "teacher-cache" / "teacher_scores.jsonl"))
    ap.add_argument("--output", default=str(ROOT / "outputs" / "checkpoints" / "boldt-modern-bi"))
    ap.add_argument("--guide-model", default=None, help="enable CachedGISTEmbedLoss with this guide")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--mini-batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--dry-run-rows", type=int, default=2000, help="rows to scan in --dry-run")
    ap.add_argument("--run-id", default=None, help="experiment run id (run card written on success)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_student_training_config(args.student_config)
    print(f"[student] base={cfg.base_model} variant={cfg.student_variant} "
          f"dims={cfg.matryoshka_dims} policy={cfg.train_eval_split_policy}")

    if not pathlib.Path(args.teacher_cache).exists():
        msg = f"teacher cache not found: {args.teacher_cache} (build it with build_teacher_cache.py)"
        if args.dry_run:
            print(f"[dry-run] WARNING: {msg}; planning loss stack only.")
            cache_rows = []
        else:
            print(f"ERROR: {msg}", file=sys.stderr)
            return 2
    else:
        limit = args.dry_run_rows if args.dry_run else None
        cache_rows = T.read_teacher_cache_jsonl(args.teacher_cache)
        if limit is not None:
            cache_rows = cache_rows[:limit]

    examples = TM.build_train_dataset_from_teacher_cache(cache_rows)
    meta = TM.dataset_metadata(examples)
    plan = TM.plan_loss_stack(cfg, meta["has_teacher_scores"], use_guide=bool(args.guide_model))
    print(f"[dataset] {json.dumps(meta, ensure_ascii=False)}")
    print(f"[loss-stack] {json.dumps(plan, ensure_ascii=False)}")

    if args.dry_run:
        assert "torch" not in sys.modules, "dry-run must not import torch"
        assert "sentence_transformers" not in sys.modules, "dry-run must not import sentence_transformers"
        print("dry-run-ok (no ML imports)")
        return 0

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"Real training needs extras: pip install -e '.[train]'. ({exc})")
    if not examples:
        print("ERROR: no training examples built from cache.", file=sys.stderr)
        return 2
    report = TM.train_modern_embedder(
        cfg, examples, args.output, epochs=args.epochs, max_steps=args.max_steps,
        batch_size=args.batch_size, mini_batch_size=args.mini_batch_size, lr=args.lr,
        bf16=args.bf16, gradient_checkpointing=args.gradient_checkpointing,
        use_lora=args.lora, guide_model_name=args.guide_model)
    card = ER.emit_run_card(args.run_id, "train_embedder", "scripts/train_modern_embedder.py",
                            model=cfg.base_model, dataset=args.teacher_cache,
                            metrics={"num_examples": report.get("num_examples")},
                            input_artifacts=[args.teacher_cache], output_artifacts=[args.output],
                            gpu=report.get("gpu_name"), notes=f"variant={cfg.student_variant}")
    report["run_card"] = card
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
