#!/usr/bin/env python3
"""Train the v6 STANDALONE German RAG reranker — evaluated RAW, with NO serving policy.

PRECONDITION (the v6 reason-for-being): first-stage candidate recall must be fixed
(`docs/first-stage-recall-audit.md`, `docs/v6-dense-rag-embedder.md`). A reranker cannot rank a
positive that is absent from the candidate list.

Trained ONLY on positive-present candidate lists. Loss: listwise KL (PRIMARY, distil
Qwen3-Reranker-8B) + pairwise margin + high-confidence-only pointwise BCE — **all restricted to
positive-present lists**. Lists where the positive is ABSENT are EXCLUDED from BCE/pairwise (their
"negatives" would be false negatives) and kept only as diagnostics. Uncertain candidates are
listwise-only. No policy loss is ever an objective, and promotion uses the RAW reranker gate.

`--dry-run` imports NO ML and writes the data report + loss plan + run card.
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
                    help="v6 candidate UNION lists, teacher-scored, with positives present")
    ap.add_argument("--reranker-config", default=str(ROOT / "configs" / "training_reranker.json"))
    ap.add_argument("--model-base", default="Boldt/Boldt-DC-350M")
    ap.add_argument("--output", default="outputs/v6-reranker/checkpoints/boldt-rag-reranker-v6")
    ap.add_argument("--loss", default="listwise_kl+pairwise+pointwise_confident")
    ap.add_argument("--min-teacher-margin", type=float, default=2.0)
    ap.add_argument("--max-pairs-per-query", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--run-id", default="v6-raw-reranker")
    ap.add_argument("--report", default=None)
    ap.add_argument("--force-research-run", action="store_true",
                    help="train even if the dense-recall STOP gate is active; marks the run "
                         "invalid_for_promotion")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # DENSE-RECALL STOP GATE: refuse to waste GPU on reranker training when first-stage recall is
    # insufficient (positives absent from candidate lists). check_dense_recall_gate writes this file.
    stop_file = ROOT / "STOP_RERANKER_TRAINING.md"
    forced_research_run = False
    if stop_file.exists():
        if not args.force_research_run:
            print(f"REFUSING to train: {stop_file.name} is active (dense recall insufficient).\n"
                  "A reranker cannot recover positives the first stage never retrieved. Fix dense "
                  "retrieval / candidate generation, re-run scripts/check_dense_recall_gate.py, then "
                  "retry. Override with --force-research-run (marks the run invalid_for_promotion).",
                  file=sys.stderr)
            return 2
        forced_research_run = True
        print("WARNING: --force-research-run set while the dense-recall STOP gate is active; this "
              "run is invalid_for_promotion.", file=sys.stderr)

    load_v5_rag_config(args.config)        # validates eval-only public benchmarks + legal-diagnostic
    rows = _read(args.candidate_lists)

    ds = RM.build_v6_reranker_dataset(rows, temperature=args.temperature,
                                      min_teacher_margin=args.min_teacher_margin,
                                      max_pairs_per_query=args.max_pairs_per_query)
    report, errors = ds["report"], ds["errors"]
    plan = RM.plan_v5_reranker_loss(args.loss)
    card = RM.v6_raw_reranker_run_card(report, plan, run_id=args.run_id, model_base=args.model_base,
                                       output=args.output, bf16=args.bf16,
                                       gradient_checkpointing=args.gradient_checkpointing,
                                       lora=args.lora)
    card["forced_research_run"] = forced_research_run
    card["invalid_for_promotion"] = forced_research_run     # a forced run can never be promoted
    out = {"run_card": card, "loss_plan": plan, "dataset_report": report, "errors": errors}

    report_path = pathlib.Path(args.report or f"outputs/v6-reranker/{args.run_id}_run_card.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    runcard_path = pathlib.Path("outputs/run-cards") / f"{args.run_id}.json"
    runcard_path.parent.mkdir(parents=True, exist_ok=True)
    runcard_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[v6-reranker] run={args.run_id} loss={plan['loss']} "
          f"present={report['lists_positive_present']} "
          f"absent_excluded={report['lists_positive_absent_excluded']} "
          f"(present_rate {report['positive_present_rate']}) -> {report_path}")
    print(f"[v6-reranker] listwise={report['listwise_batches']} pairwise={report['pairwise_pairs']} "
          f"pointwise={report['pointwise_examples']} margin={report['teacher_margin']['avg']} "
          f"domains={report['examples_by_domain']}")
    for e in errors[:5]:
        print(f"  ✗ {e}", file=sys.stderr)
    if errors:
        print(f"FAIL — {len(errors)} leakage error(s); fail closed", file=sys.stderr)
        return 1
    if report["lists_positive_present"] == 0:
        print("FAIL — no positive-present lists; nothing valid to train on", file=sys.stderr)
        return 1

    if args.dry_run:
        assert "torch" not in sys.modules, "dry-run must not import torch"
        print("dry-run-ok (no ML imports; data report + loss plan + run card written, no training)")
        return 0

    # ---- real training (lazy ML): listwise KL primary + pairwise + pointwise, present-only ----
    from boldt_embed.config import load_reranker_config
    cfg = load_reranker_config(args.reranker_config)
    cfg.model_name_or_path = args.model_base
    result = RM.train_v6_raw_reranker(
        cfg, ds["listwise"], args.output, temperature=args.temperature,
        min_teacher_margin=args.min_teacher_margin, max_steps=args.max_steps, bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing, use_lora=args.lora)
    card["training_result"] = result
    out["run_card"] = card
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    runcard_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v6-reranker] trained raw (listwise primary; {result.get('pairwise_pairs_used')} pairwise, "
          f"{result.get('pointwise_examples_used')} pointwise) -> {args.output}")
    print("[v6-reranker] NEXT: evaluate RAW lift over fixed candidate lists (no serving policy).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
