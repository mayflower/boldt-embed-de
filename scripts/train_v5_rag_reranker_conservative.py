#!/usr/bin/env python3
"""Train the CONSERVATIVE v5 RAG reranker: listwise-KL primary + a rank-preservation penalty on
high-first-stage-confidence lists, to cut near-ceiling churn (the v5 / abstain failure mode).

Total loss = listwise_teacher_kl + pairwise_margin + pointwise_confident_bce
             + lambda_preserve * rank_preservation_loss   (preservation on high-confidence lists only)

Uses the EXISTING v5 teacher-scored candidate lists (no new teacher calls). FAQ-share-capped,
domain-balanced. `--dry-run` imports no torch (writes the loss plan + run card).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import rank_preservation_loss as RP  # noqa: E402
from boldt_embed import reranker_modern as RM  # noqa: E402
from boldt_embed.v5_rag_config import load_v5_rag_config  # noqa: E402


def _read(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiments" / "v5_small_rag.json"))
    ap.add_argument("--candidate-lists", required=True)
    ap.add_argument("--reranker-config", default=str(ROOT / "configs" / "training_reranker.json"))
    ap.add_argument("--model-base", default="Boldt/Boldt-DC-350M")
    ap.add_argument("--output", default="outputs/v5-small-rag/checkpoints/boldt-rag-reranker-v5-conservative")
    ap.add_argument("--lambda-preserve", type=float, default=0.2)
    ap.add_argument("--justify-margin", type=float, default=RP.DEFAULT_JUSTIFY_MARGIN)
    ap.add_argument("--teacher-margin-override", type=float, default=None,
                    help="teacher advantage that justifies an inversion (overrides --justify-margin)")
    ap.add_argument("--fs-gap-percentile", type=float, default=RP.DEFAULT_FS_GAP_PERCENTILE)
    ap.add_argument("--high-confidence-percentile", type=float, default=None,
                    help="first-stage-gap percentile defining high-confidence lists (overrides --fs-gap-percentile)")
    ap.add_argument("--preserve-top-k", type=int, default=None,
                    help="protect only the first-stage top-k from teacher-unjustified demotion")
    ap.add_argument("--max-faq-share", type=float, default=0.35)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--run-id", default="v5-reranker-conservative")
    ap.add_argument("--report", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_v5_rag_config(args.config)
    rows = _read(args.candidate_lists)
    cap = RM.cap_faq_share(rows, args.max_faq_share)
    kept = cap["kept"]
    report = RM.v5_reranker_training_report(kept)
    plan = RP.plan_conservative_loss(args.lambda_preserve)
    hc_pct = args.high_confidence_percentile if args.high_confidence_percentile is not None \
        else args.fs_gap_percentile
    justify = args.teacher_margin_override if args.teacher_margin_override is not None \
        else args.justify_margin
    batches = RP.scored_lists_to_conservative_batches(kept, fs_gap_percentile=hc_pct)
    n_high = sum(1 for b in batches if b["high_confidence"])
    disp = RP.displacement_proxy(batches)

    card = RM.v5_reranker_run_card(report, {"loss": "conservative", "components": plan["components"],
                                            "primary": "listwise", "listwise_variant": "listwise_kl",
                                            "uncertain_listwise_only": True,
                                            "matryoshka_dims": []}, cap,
                                   run_id=args.run_id, model_base=args.model_base,
                                   output=args.output, bf16=args.bf16,
                                   gradient_checkpointing=args.gradient_checkpointing)
    card["conservative"] = {"lambda_preserve": args.lambda_preserve,
                            "justify_margin": justify, "teacher_margin_override": args.teacher_margin_override,
                            "high_confidence_percentile": hc_pct, "preserve_top_k": args.preserve_top_k,
                            "high_confidence_lists": n_high, "total_lists": len(batches),
                            "high_confidence_share": round(n_high / max(len(batches), 1), 4),
                            "expected_max_displacement_proxy": disp}
    cap_summary = {k: v for k, v in cap.items() if k != "kept"}
    out = {"run_card": card, "loss_plan": plan, "faq_cap": cap_summary, "training_report": report}

    report_path = pathlib.Path(args.report or f"outputs/v5-small-rag/{args.run_id}_run_card.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    pathlib.Path("outputs/run-cards").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"outputs/run-cards/{args.run_id}.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[v5-conservative] lambda_preserve={args.lambda_preserve} hc_pct={hc_pct} "
          f"preserve_top_k={args.preserve_top_k} justify={justify} "
          f"high_confidence_lists={n_high}/{len(batches)} "
          f"({card['conservative']['high_confidence_share']}) "
          f"teacher_disp_proxy={disp['mean_teacher_max_displacement']} faq_share={report['faq_share']} "
          f"not_faq_only={report['not_faq_only']} -> {report_path}")
    if cap["status"] != "pass" or not report["not_faq_only"]:
        print("FAIL — FAQ-only training data", file=sys.stderr)
        return 1
    if n_high == 0:
        print("WARNING: 0 high-confidence lists detected — preservation term would be a no-op "
              "(check first_stage_score presence / fs-gap-percentile)", file=sys.stderr)

    if args.dry_run:
        assert "torch" not in sys.modules, "dry-run must not import torch"
        print("dry-run-ok (no ML imports; loss plan + run card written, no training)")
        return 0

    from boldt_embed.config import load_reranker_config
    cfg = load_reranker_config(args.reranker_config)
    cfg.model_name_or_path = args.model_base
    result = RM.train_conservative_listwise_reranker(
        cfg, batches, args.output, lambda_preserve=args.lambda_preserve,
        justify_margin=justify, preserve_top_k=args.preserve_top_k, temperature=args.temperature,
        bf16=args.bf16, gradient_checkpointing=args.gradient_checkpointing)
    card["training_result"] = result
    out["run_card"] = card
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    pathlib.Path(f"outputs/run-cards/{args.run_id}.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v5-conservative] trained ({result.get('high_confidence_lists')} high-conf lists; "
          f"mean_listwise={result.get('mean_listwise_loss')} "
          f"mean_preservation={result.get('mean_preservation_loss')}) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
