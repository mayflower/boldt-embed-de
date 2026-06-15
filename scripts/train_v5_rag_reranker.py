#!/usr/bin/env python3
"""Train the v5 German RAG reranker that does NOT overfit to WebFAQ.

Listwise KL is the PRIMARY objective (distill the Qwen3-Reranker-8B candidate distribution), with
pairwise (RankNet/margin, strong teacher margins only) and high-confidence-only pointwise BCE as
auxiliaries; UNCERTAIN candidates are listwise-only (never a hard BCE label). The training mix is
**FAQ-share-capped** and domain/source-balanced so it is demonstrably not FAQ-only. Evaluation is
the hardness-aware gate (`scripts/eval_v5_rag_lift.py`), not raw WebFAQ lift.

`--dry-run` imports NO ML and writes the FAQ-cap report + loss plan + run card.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import reranker_modern as RM  # noqa: E402  (stdlib funcs; ML lazy inside)
from boldt_embed.v5_rag_config import load_v5_rag_config  # noqa: E402


def _read(path):
    p = pathlib.Path(path)
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").split("\n") if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiments" / "v5_small_rag.json"))
    ap.add_argument("--candidate-lists", required=True,
                    help="v5 teacher-scored candidate lists (multi-domain)")
    ap.add_argument("--reranker-config", default=str(ROOT / "configs" / "training_reranker.json"))
    ap.add_argument("--model-base", default="Boldt/Boldt-DC-350M")
    ap.add_argument("--output", default="outputs/v5-small-rag/checkpoints/boldt-rag-reranker-v5")
    ap.add_argument("--loss", default="listwise_kl+pairwise+pointwise_confident")
    ap.add_argument("--max-faq-share", type=float, default=0.35)
    ap.add_argument("--max-per-domain", type=int, default=None,
                    help="cap lists per domain for balanced batches")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lora", action="store_true", help="LoRA-tune (e.g. Qwen3-Reranker-0.6B)")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--run-id", default="v5-reranker-boldt")
    ap.add_argument("--report", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_v5_rag_config(args.config)        # validates the v5 experiment (eval-only public, legal diag)
    rows = _read(args.candidate_lists)

    cap = RM.cap_faq_share(rows, args.max_faq_share)
    kept = cap["kept"]
    if args.max_per_domain:
        kept = RM.domain_balanced_list_sampler(kept, max_per_domain=args.max_per_domain)
    report = RM.v5_reranker_training_report(kept)
    plan = RM.plan_v5_reranker_loss(args.loss)
    card = RM.v5_reranker_run_card(report, plan, cap, run_id=args.run_id, model_base=args.model_base,
                                   output=args.output, bf16=args.bf16,
                                   gradient_checkpointing=args.gradient_checkpointing, lora=args.lora)
    cap_summary = {k: v for k, v in cap.items() if k != "kept"}   # never embed the full lists
    out = {"run_card": card, "loss_plan": plan, "faq_cap": cap_summary, "training_report": report}

    report_path = pathlib.Path(args.report or f"outputs/v5-small-rag/{args.run_id}_run_card.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    runcard_path = pathlib.Path("outputs/run-cards") / f"{args.run_id}.json"
    runcard_path.parent.mkdir(parents=True, exist_ok=True)
    runcard_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[v5-reranker] run={args.run_id} loss={plan['loss']} faq_share={report['faq_share']} "
          f"(cap {args.max_faq_share}; was {cap['faq_share_before']}, dropped {cap['faq_dropped_for_cap']}) "
          f"non_faq={report['nonfaq_share']} not_faq_only={report['not_faq_only']} -> {report_path}")
    print(f"[v5-reranker] domains={report['examples_by_domain']} "
          f"uncertain_fraction={report['uncertain_fraction']}")

    if cap["status"] != "pass" or not report["not_faq_only"]:
        print(f"FAIL — FAQ-only training data (no non-FAQ lists): {cap.get('reason', '')}",
              file=sys.stderr)
        return 1

    if args.dry_run:
        assert "torch" not in sys.modules, "dry-run must not import torch"
        print("dry-run-ok (no ML imports; FAQ-cap report + loss plan + run card written)")
        return 0

    # ---- real training (lazy ML): listwise KL primary via the distillation trainer ----
    from boldt_embed.config import load_reranker_config
    cfg = load_reranker_config(args.reranker_config)
    cfg.model_name_or_path = args.model_base
    batches = RM.scored_lists_to_listwise(kept, temperature=args.temperature)
    result = RM.train_listwise_distilled_reranker(
        cfg, batches, args.output, bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing, use_lora=args.lora,
        temperature=args.temperature)
    card["training_result"] = result
    out["run_card"] = card
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    runcard_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v5-reranker] trained (listwise primary, {result.get('num_queries')} queries) "
          f"-> {args.output}; run card -> {runcard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
