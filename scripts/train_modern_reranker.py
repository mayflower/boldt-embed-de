#!/usr/bin/env python3
"""Train the Boldt reranker with pointwise / pairwise / listwise / mixed objectives (Prompt 7).

Input is the teacher cache and/or mined hard negatives. `--dry-run` builds the examples for
the chosen objective and prints counts WITHOUT importing torch. Real training needs the
`train` extras + a GPU. The legacy BCE reranker (`train_reranker_de.py`) remains a baseline.
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
from boldt_embed import teacher as T  # noqa: E402
from boldt_embed.config import load_reranker_config  # noqa: E402


def _cache_rows_from_inputs(teacher_cache, hard_negatives):
    """Normalize teacher-cache rows and/or hard-negative rows into per-(query,doc) cache rows."""
    rows = []
    if teacher_cache and pathlib.Path(teacher_cache).exists():
        rows.extend(T.read_teacher_cache_jsonl(teacher_cache))
    if hard_negatives and pathlib.Path(hard_negatives).exists():
        for r in dp.stream_jsonl(hard_negatives):
            qid, q = str(r["query_id"]), r.get("query", "")
            rows.append({"query_id": qid, "doc_id": r["positive_doc_id"], "query": q,
                         "document": r["positive"], "positive": True})
            for n in r.get("negatives", []):
                rows.append({"query_id": qid, "doc_id": n["doc_id"], "query": q,
                             "document": n["document"], "positive": False,
                             "reranker_score": n.get("reranker_teacher_score"),
                             "embedding_score": n.get("embedding_teacher_score")})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "training_reranker.json"))
    ap.add_argument("--teacher-cache", default=None)
    ap.add_argument("--hard-negatives", default=None)
    ap.add_argument("--output", default=str(ROOT / "outputs" / "checkpoints" / "boldt-reranker-modern"))
    ap.add_argument("--loss", choices=["pointwise", "pairwise", "listwise", "mixed"], default="pointwise")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--run-id", default=None, help="experiment run id (run card written on success)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_reranker_config(args.config)
    rows = _cache_rows_from_inputs(args.teacher_cache, args.hard_negatives)
    if not rows:
        msg = "no input rows (provide --teacher-cache and/or --hard-negatives)"
        print(f"{'[dry-run] ' if args.dry_run else 'ERROR: '}{msg}",
              file=sys.stderr if not args.dry_run else sys.stdout)
        if not args.dry_run:
            return 2

    pointwise = RM.build_reranker_examples_from_teacher_cache(rows)
    pairwise = RM.build_pairwise_examples(rows)
    listwise = RM.build_listwise_batches(rows, temperature=args.temperature)
    plan = {"objective": args.loss, "base_model": cfg.model_name_or_path,
            "pointwise_examples": len(pointwise), "pairwise_examples": len(pairwise),
            "listwise_queries": len(listwise)}
    print(f"[plan] {json.dumps(plan, ensure_ascii=False)}")

    if args.dry_run:
        assert "torch" not in sys.modules, "dry-run must not import torch"
        print("dry-run-ok (no ML imports)")
        return 0

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"Real training needs extras: pip install -e '.[train]'. ({exc})")

    reports = []
    if args.loss in ("pointwise", "mixed"):
        reports.append(RM.train_pointwise_reranker(
            cfg, pointwise, args.output, epochs=args.epochs, batch_size=args.batch_size,
            max_length=args.max_length, lr=args.lr, bf16=args.bf16, use_lora=args.lora))
    if args.loss in ("pairwise", "mixed"):
        reports.append(RM.train_pairwise_reranker(
            cfg, pairwise, args.output, epochs=args.epochs, batch_size=args.batch_size,
            max_length=args.max_length, lr=args.lr, margin=args.margin, bf16=args.bf16,
            use_lora=args.lora))
    if args.loss in ("listwise", "mixed"):
        reports.append(RM.train_listwise_distilled_reranker(
            cfg, listwise, args.output, epochs=args.epochs, max_length=args.max_length,
            lr=args.lr, temperature=args.temperature, bf16=args.bf16, use_lora=args.lora))
    card = ER.emit_run_card(args.run_id, "train_reranker", "scripts/train_modern_reranker.py",
                            model=cfg.model_name_or_path,
                            dataset=args.teacher_cache or args.hard_negatives,
                            metrics={"objective": args.loss,
                                     "final_loss": reports[-1].get("final_loss") if reports else None},
                            input_artifacts=[p for p in (args.teacher_cache, args.hard_negatives) if p],
                            output_artifacts=[args.output], notes=f"loss={args.loss}")
    print(json.dumps({"trained": reports, "run_card": card}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
