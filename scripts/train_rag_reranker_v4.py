#!/usr/bin/env python3
"""Train the v4 German RAG reranker with LISTWISE-primary candidate-list supervision.

Input is teacher-scored candidate lists (`rag_teacher_scoring`). The objective is **listwise**
distillation over the teacher_score distribution per top-k list; pairwise margin reinforces
gold > hard-negative on strong teacher margins; pointwise BCE is restricted to high-confidence
gold + clear hard negatives so it cannot dominate on noisy labels; optional MSE regresses the
teacher score. Domain/source-balanced sampling. `--dry-run` imports no torch.

Never trains on GermanQuAD/DT-test eval labels or the WebFAQ held-out split: the input is the
train split, and `--eval-query-ids` (optional) is a hard guard against any eval query leaking in.
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
from boldt_embed import reranker_modern as RM  # noqa: E402
from boldt_embed.config import load_reranker_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-lists", required=True, help="teacher-scored RAG lists JSONL")
    ap.add_argument("--config", default=str(ROOT / "configs" / "training_reranker.json"))
    ap.add_argument("--output", default=str(ROOT / "outputs" / "v4-rag-reranker" / "checkpoints" / "boldt-rag-reranker-v4"))
    ap.add_argument("--loss", choices=["listwise", "mixed_listwise"], default="mixed_listwise")
    ap.add_argument("--with-mse", action="store_true", help="add MSE regression to teacher_score")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--pairwise-min-teacher-margin", type=float, default=2.0)
    ap.add_argument("--max-lists-per-domain", type=int, default=None)
    ap.add_argument("--max-lists-per-source", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-query-ids", default=None,
                    help="optional JSONL/txt of held-out eval query_ids that MUST NOT appear in training")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not pathlib.Path(args.candidate_lists).exists():
        print(f"ERROR: candidate lists not found: {args.candidate_lists}", file=sys.stderr)
        return 2
    rows = list(dp.stream_jsonl(args.candidate_lists))

    # leakage guard: no eval query may appear in training
    if args.eval_query_ids and pathlib.Path(args.eval_query_ids).exists():
        eval_ids = set()
        for line in pathlib.Path(args.eval_query_ids).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                eval_ids.add(str(json.loads(line).get("query_id")))
            except json.JSONDecodeError:
                eval_ids.add(line)
        leaked = sorted({str(r.get("query_id")) for r in rows} & eval_ids)
        if leaked:
            print(f"ERROR: {len(leaked)} eval query_id(s) present in training data: {leaked[:5]}",
                  file=sys.stderr)
            return 2

    rows = RM.domain_balanced_list_sampler(rows, max_per_domain=args.max_lists_per_domain,
                                           max_per_source=args.max_lists_per_source, seed=args.seed)

    listwise = RM.scored_lists_to_listwise(rows, temperature=args.temperature)
    pairwise = RM.candidate_lists_to_pairwise(rows, min_teacher_margin=args.pairwise_min_teacher_margin)
    pointwise = RM.scored_lists_to_pointwise_high_confidence(rows)
    mse = RM.scored_lists_to_mse(rows) if args.with_mse else []

    cfg = load_reranker_config(args.config)
    plan = RM.plan_rag_reranker_loss(args.loss, with_mse=args.with_mse)
    report = RM.rag_reranker_training_report(rows)
    plan_out = {"objective": args.loss, "loss": plan, "base_model": cfg.model_name_or_path,
                "input": f"{args.candidate_lists} ({len(rows)} lists)",
                "listwise_queries": len(listwise), "pairwise_examples": len(pairwise),
                "pointwise_high_confidence_examples": len(pointwise), "mse_examples": len(mse),
                "training_report": report}
    print(f"[plan] {json.dumps(plan_out, ensure_ascii=False)}")

    if args.dry_run:
        assert "torch" not in sys.modules, "dry-run must not import torch"
        print("dry-run-ok (no ML imports)")
        return 0

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"Real training needs extras: pip install -e '.[train]'. ({exc})")

    reports = []
    # listwise FIRST (primary), then pairwise, then the restricted pointwise BCE, then optional MSE.
    reports.append(RM.train_listwise_distilled_reranker(
        cfg, listwise, args.output, epochs=args.epochs, max_length=args.max_length, lr=args.lr,
        temperature=args.temperature, bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing, use_lora=args.lora))
    if args.loss == "mixed_listwise":
        if pairwise:
            reports.append(RM.train_pairwise_reranker(
                cfg, pairwise, args.output, epochs=args.epochs, batch_size=args.batch_size,
                max_length=args.max_length, lr=args.lr, margin=args.margin, bf16=args.bf16,
                gradient_checkpointing=args.gradient_checkpointing, use_lora=args.lora))
        if pointwise:
            reports.append(RM.train_pointwise_reranker(
                cfg, pointwise, args.output, epochs=args.epochs, batch_size=args.batch_size,
                max_length=args.max_length, lr=args.lr, bf16=args.bf16,
                gradient_checkpointing=args.gradient_checkpointing, use_lora=args.lora))
        if mse:
            reports.append(RM.train_pointwise_reranker(
                cfg, mse, args.output, epochs=args.epochs, batch_size=args.batch_size,
                max_length=args.max_length, lr=args.lr, regression=True, bf16=args.bf16,
                gradient_checkpointing=args.gradient_checkpointing, use_lora=args.lora))

    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output, "rag_reranker_training_report.json").write_text(
        json.dumps({"plan": plan_out, "trained": reports}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    card = ER.emit_run_card(args.run_id, "train_reranker", "scripts/train_rag_reranker_v4.py",
                            model=cfg.model_name_or_path, dataset=args.candidate_lists,
                            metrics={"objective": args.loss, "lists": report["lists"],
                                     "gold_positives": report["gold_positives"]},
                            input_artifacts=[args.candidate_lists], output_artifacts=[args.output],
                            notes=f"v4 RAG reranker loss={args.loss}")
    print(json.dumps({"trained": reports, "run_card": card}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
