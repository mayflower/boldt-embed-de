#!/usr/bin/env python3
"""Train dense-v6.1 to improve WebFAQ Recall@50 while preserving Recall@100 + guardrails.

Continues from the dense-v6 checkpoint. Loss: CachedMultipleNegativesRankingLoss -> MatryoshkaLoss
[1024..64], with a RANK-PROMOTION objective realized as CMNRL over (query, positive, top50-blocker)
triplets — for queries whose positive sits at dense rank 51..200, the positive is pushed above the
docs that currently outrank it. The triplets' negatives are teacher-vetted (mining veto). MarginMSE
is teacher-margin-prepared at the data level. NO_DUPLICATES sampler. **DENSE-ONLY — no reranker.**

`--dry-run` imports NO ML and writes the dataset report + loss plan + run card.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import train_modern as TM  # noqa: E402  (stdlib funcs; ML lazy inside)
from boldt_embed.v6_1_dense_config import load_v6_1_dense_config  # noqa: E402


def _read(path):
    p = pathlib.Path(path)
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").split("\n") if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiments" / "v6_1_dense_top50.json"))
    ap.add_argument("--base-model", default="outputs/v6-dense-rag/checkpoints/boldt-dense-rag-v6")
    ap.add_argument("--train-pairs", default="data/processed/v6/rag_pairs_teacher_validated.jsonl")
    ap.add_argument("--hard-negatives", default="data/processed/v6_1/dense_top50_hardnegatives.jsonl")
    ap.add_argument("--output", default="outputs/v6-1-dense-top50/checkpoints/boldt-dense-rag-v6-1")
    ap.add_argument("--max-triplets-per-query", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--grad-accumulation", type=int, default=1,
                    help="gradient accumulation steps; effective batch = batch-size * this")
    ap.add_argument("--mini-batch-size", type=int, default=None,
                    help="CachedMNRL/GradCache chunk size (caps activation memory of the cached forward)")
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=None,
                    help="contrastive temperature (CMNRL scale=1/temperature); None=SBERT default")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--bidirectional", action="store_true",
                    help="v8: convert the causal decoder to bidirectional attention (LLM2Vec mask "
                         "patch, eager attention). Eval MUST re-apply the patch (runner --bidirectional).")
    ap.add_argument("--run-id", default="v6-1-dense-top50")
    ap.add_argument("--report", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_v6_1_dense_config(args.config)         # fail-closed; reranker_training_enabled=false
    assert cfg["reranker_training_enabled"] is False, "v6.1 is dense-only — no reranker training"

    pairs = _read(args.train_pairs)
    hardnegs = _read(args.hard_negatives)
    ds = TM.build_v6_1_dense_dataset(pairs, hardnegs,
                                     max_triplets_per_query=args.max_triplets_per_query)
    report, errors = ds["report"], ds["errors"]
    plan = TM.plan_v6_1_loss_stack(has_teacher_margins=report["has_teacher_margins"],
                                   matryoshka_dims=TM.MATRYOSHKA_DEFAULT)
    card = TM.v6_1_dense_run_card(report, plan, run_id=args.run_id, base_model=args.base_model,
                                  output=args.output, max_steps=args.max_steps,
                                  batch_size=args.batch_size, bf16=args.bf16,
                                  gradient_checkpointing=args.gradient_checkpointing)
    optimizer = {"lr": args.lr, "warmup_ratio": args.warmup_ratio, "temperature": args.temperature}
    card["optimizer"] = optimizer
    out = {"run_card": card, "loss_plan": plan, "dataset_report": report, "errors": errors,
           "optimizer": optimizer}

    report_path = pathlib.Path(args.report or f"outputs/v6-1-dense-top50/{args.run_id}_run_card.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    runcard_path = pathlib.Path("outputs/run-cards") / f"{args.run_id}.json"
    runcard_path.parent.mkdir(parents=True, exist_ok=True)
    runcard_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[v6.1-dense] run={args.run_id} base={args.base_model} pairs={report['pair_examples']} "
          f"rank_promotion_triplets={report['rank_promotion_triplets']} "
          f"(queries rank51-100={report['positive_rank_51_100']}, "
          f"rank101-200={report['positive_rank_101_200']}) loss={plan['loss_stack']} -> {report_path}")
    print(f"[v6.1-dense] domain_mix={report['domain_mix']} margins={report['hard_negative_margins']}")
    for e in errors[:5]:
        print(f"  ✗ {e}", file=sys.stderr)
    if errors:
        print(f"FAIL — {len(errors)} leakage error(s); fail closed", file=sys.stderr)
        return 1
    if report["rank_promotion_triplets"] == 0:
        print("FAIL — no rank-promotion triplets (no rank-51..200 cases)", file=sys.stderr)
        return 1

    if args.dry_run:
        assert "torch" not in sys.modules, "dry-run must not import torch"
        print("dry-run-ok (no ML; dataset report + loss plan + run card written, no training)")
        return 0

    # ---- real training (lazy ML): CMNRL+Matryoshka over pairs + rank-promotion triplets ----
    result = TM.train_v6_1_dense_embedder(
        args.base_model, ds["pair_examples"], ds["triplet_examples"], args.output,
        matryoshka_dims=plan["matryoshka_dims"], max_steps=args.max_steps,
        batch_size=args.batch_size, lr=args.lr, warmup_ratio=args.warmup_ratio,
        temperature=args.temperature, max_seq_length=args.max_seq_length, bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing, bidirectional=args.bidirectional,
        gradient_accumulation=args.grad_accumulation, mini_batch_size=args.mini_batch_size)
    card["bidirectional"] = args.bidirectional
    card["training_result"] = result
    out["run_card"] = card
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    runcard_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v6.1-dense] trained -> {args.output}")
    print("[v6.1-dense] NEXT: re-run the recall audit + dense-recall gate (target Recall@50 >= 0.90). "
          "Reranker work stays deferred until dense-v6.1 is evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
