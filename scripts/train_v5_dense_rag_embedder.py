#!/usr/bin/env python3
"""Train the v5 small German DENSE RAG retriever (not another cross-encoder).

A better dense retriever lifts first-stage recall and produces better candidate lists for the
reranker. Loss stack: CachedMultipleNegativesRankingLoss -> MatryoshkaLoss[1024,768,512,256,128,64]
+ MarginMSELoss (Qwen3-8B teacher scores for hard negatives) + optional EmbedDistillLoss (Qwen3-
Embedding-8B vectors), NO_DUPLICATES sampler. Models: Boldt causal v5, or Qwen3-Embedding-0.6B LoRA.

Fails closed on public-benchmark/eval leakage; synthetic pairs train only when teacher-validated.
`--dry-run` imports NO ML and writes the loss plan + run card.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import train_modern as TM  # noqa: E402  (stdlib funcs; ML lazy inside)
from boldt_embed.v5_rag_config import load_v5_rag_config  # noqa: E402

DEFAULT_BOLDT = "outputs/v3-real-domain/checkpoints/boldt-modern-causal-v3"


def _read(path):
    p = pathlib.Path(path)
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").split("\n") if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiments" / "v5_small_rag.json"))
    ap.add_argument("--train-pairs", required=True)
    ap.add_argument("--hard-negatives", default=None)
    ap.add_argument("--teacher-scores", default=None,
                    help="outputs/v5-small-rag/teacher/rag_embedder_teacher_scores.jsonl")
    ap.add_argument("--distill-vectors", default=None,
                    help="Qwen3-Embedding-8B teacher vectors JSONL (enables EmbedDistillLoss)")
    ap.add_argument("--output", default="outputs/v5-small-rag/checkpoints/boldt-dense-v5")
    ap.add_argument("--model", default=DEFAULT_BOLDT, help="Boldt causal v5 base, or a Qwen3 model")
    ap.add_argument("--lora", action="store_true", help="LoRA-tune (e.g. Qwen3-Embedding-0.6B)")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--teacher-threshold", type=float, default=4.0)
    ap.add_argument("--run-id", default="v5-dense-boldt")
    ap.add_argument("--report", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_v5_rag_config(args.config)        # validates (public-benchmarks-eval-only, legal diagnostic)

    pairs = _read(args.train_pairs)
    hardnegs = _read(args.hard_negatives) if args.hard_negatives else []
    teacher_scores = _read(args.teacher_scores) if args.teacher_scores else []

    ds = TM.build_v5_dense_dataset(pairs, hardnegs, teacher_scores,
                                   teacher_threshold=args.teacher_threshold)
    report, errors = ds["report"], ds["errors"]
    has_distill = report["has_distill_vectors"] or bool(args.distill_vectors)
    plan = TM.plan_v5_dense_loss_stack(has_teacher_scores=report["has_teacher_scores"],
                                       has_distill_vectors=has_distill)
    card = TM.v5_dense_run_card(report, plan, run_id=args.run_id, model=args.model,
                                output=args.output, max_steps=args.max_steps, bf16=args.bf16,
                                gradient_checkpointing=args.gradient_checkpointing, lora=args.lora)
    out_report = {"run_card": card, "loss_plan": plan, "dataset_report": report, "errors": errors}

    report_path = pathlib.Path(args.report or f"outputs/v5-small-rag/{args.run_id}_run_card.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out_report, ensure_ascii=False, indent=2), encoding="utf-8")
    runcard_path = pathlib.Path("outputs/run-cards") / f"{args.run_id}.json"
    runcard_path.parent.mkdir(parents=True, exist_ok=True)
    runcard_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[v5-dense] run={args.run_id} model={args.model} examples={report['examples']} "
          f"loss={plan['loss_stack']} sampler={plan['batch_sampler']} -> {report_path}")
    print(f"[v5-dense] domain_mix={report['domain_mix']} "
          f"teacher_validation={report['teacher_validation']}")
    for e in errors[:5]:
        print(f"  ✗ {e}", file=sys.stderr)
    if errors:
        print(f"FAIL — {len(errors)} leakage/validation error(s); fail closed", file=sys.stderr)
        return 1

    if args.dry_run:
        assert "torch" not in sys.modules, "dry-run must not import torch"
        print("dry-run-ok (no ML imports; loss plan + run card written, no training)")
        return 0

    # ---- real training (lazy ML) ----
    from types import SimpleNamespace
    losses = ["matryoshka"] + (["margin_mse"] if report["has_teacher_scores"] else [])
    student_cfg = SimpleNamespace(base_model=args.model, student_variant="causal",
                                  losses=losses, matryoshka_dims=plan["matryoshka_dims"])
    result = TM.train_modern_embedder(
        student_cfg, ds["examples"], args.output, max_steps=args.max_steps,
        batch_size=args.batch_size, bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing, use_lora=args.lora,
        base_model=args.model, extra_report={"run_card": card})
    card["training_result"] = result
    report_path.write_text(json.dumps({"run_card": card, "loss_plan": plan,
                                        "dataset_report": report}, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    runcard_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v5-dense] trained -> {args.output}; run card -> {runcard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
