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
    ap.add_argument("--base-model", default=None,
                    help="override student_cfg.base_model (e.g. an MNTP-adapted checkpoint)")
    ap.add_argument("--hard-negatives", default=None,
                    help="mined hard-negative JSONL (triplet training); overrides cache dataset")
    ap.add_argument("--bidirectional", choices=["auto", "true", "false"], default="auto",
                    help="override the student variant (auto = from config)")
    ap.add_argument("--use-teacher-score-distillation", choices=["auto", "true", "false"],
                    default="auto")
    ap.add_argument("--effective-batch-size", type=int, default=None,
                    help="logical contrastive batch via cached loss (informational)")
    ap.add_argument("--run-id", default=None, help="experiment run id (run card written on success)")
    ap.add_argument("--require-leakage-report", default=None,
                    help="path to a full leakage report (scripts/run_full_leakage_scan.py); refuse "
                         "to train unless it exists and is clean or cleaned (v3 gate)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # v3 gate: do not train on candidates that have not passed a full, clean leakage scan.
    if args.require_leakage_report:
        from boldt_embed.leakage_index import require_clean_leakage_report
        require_clean_leakage_report(args.require_leakage_report)  # raises ValueError if not clean
        print(f"[leakage] report OK: {args.require_leakage_report}")

    cfg = load_student_training_config(args.student_config)
    if args.base_model:
        cfg.base_model = args.base_model  # train contrastive on an MNTP-adapted checkpoint
    bidi = {"auto": None, "true": True, "false": False}[args.bidirectional]
    distill = {"auto": None, "true": True, "false": False}[args.use_teacher_score_distillation]
    print(f"[student] base={cfg.base_model} variant={cfg.student_variant} "
          f"bidirectional={bidi if bidi is not None else 'auto'} dims={cfg.matryoshka_dims} "
          f"policy={cfg.train_eval_split_policy}")

    # Dataset: prefer the mined hard-negative file (triplets) when given; else the teacher cache.
    if args.hard_negatives and pathlib.Path(args.hard_negatives).exists():
        hn_rows = list(T.stream_jsonl(args.hard_negatives))
        if args.dry_run:
            hn_rows = hn_rows[:args.dry_run_rows]
        examples = TM.build_train_dataset_from_hardneg(hn_rows)
        print(f"[data] hard-negatives: {args.hard_negatives} -> {len(examples)} triplet examples")
    else:
        if not pathlib.Path(args.teacher_cache).exists():
            msg = f"teacher cache not found: {args.teacher_cache} (build it with build_teacher_cache.py)"
            if args.dry_run:
                print(f"[dry-run] WARNING: {msg}; planning loss stack only.")
                cache_rows = []
            else:
                print(f"ERROR: {msg}", file=sys.stderr)
                return 2
        else:
            cache_rows = T.read_teacher_cache_jsonl(args.teacher_cache)
            if args.dry_run:
                cache_rows = cache_rows[:args.dry_run_rows]
        examples = TM.build_train_dataset_from_teacher_cache(cache_rows)

    meta = TM.dataset_metadata(examples)
    bidi_eff = bidi if bidi is not None else (cfg.student_variant == "bidirectional")
    plan = TM.plan_loss_stack(cfg, meta["has_teacher_scores"], use_guide=bool(args.guide_model),
                              use_distillation=distill)
    reg_plan = TM.plan_edge_spectrum_regularizer(cfg.raw.get("edge_spectrum_regularizer"))
    print(f"[dataset] {json.dumps(meta, ensure_ascii=False)}")
    print(f"[loss-stack] {json.dumps(plan, ensure_ascii=False)} bidirectional={bidi_eff}")
    print(f"[edge-spectrum-regularizer] {json.dumps(reg_plan, ensure_ascii=False)}")

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
        print("ERROR: no training examples built.", file=sys.stderr)
        return 2
    report = TM.train_modern_embedder(
        cfg, examples, args.output, epochs=args.epochs, max_steps=args.max_steps,
        batch_size=args.batch_size, mini_batch_size=args.mini_batch_size, lr=args.lr,
        bf16=args.bf16, gradient_checkpointing=args.gradient_checkpointing,
        use_lora=args.lora, guide_model_name=args.guide_model, bidirectional=bidi,
        use_distillation=distill, edge_reg=cfg.raw.get("edge_spectrum_regularizer"),
        extra_report={"hard_negatives": args.hard_negatives, "teacher_cache": args.teacher_cache})
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
