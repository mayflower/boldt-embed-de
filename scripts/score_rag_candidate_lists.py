#!/usr/bin/env python3
"""Teacher-score full RAG candidate lists (not only positive pairs).

Scores every (query, document) candidate with Qwen3-Reranker-8B (and optionally Qwen3-Embedding
cosine), then annotates each list with teacher_rank, a listwise softmax target, and the
high-precision label policy (gold positive / teacher-only-positive=uncertain / hard negative /
uncertain). `--dry-run` does NOT import torch: it annotates from any teacher_score already on the
candidates (or just plans) and writes the summary — for CI / wiring checks.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import rag_teacher_scoring as RT  # noqa: E402


def _read(path):
    return [json.loads(l) for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _has_all_scores(rows):
    return rows and all(c.get("teacher_score") is not None
                        for r in rows for c in r.get("candidates", []))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="RAG candidate-list JSONL")
    ap.add_argument("--teacher-config", default=str(ROOT / "configs" / "teacher_models.json"))
    ap.add_argument("--mode", choices=["reranker", "both"], default="reranker")
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--positive-threshold", type=float, default=RT.POSITIVE_THRESHOLD)
    ap.add_argument("--hard-neg-margin", type=float, default=RT.HARD_NEG_MARGIN)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--use-teacher-only-positives", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = _read(args.input)
    print(f"[plan] {len(rows)} lists, "
          f"{sum(len(r.get('candidates', [])) for r in rows)} candidate pairs to score "
          f"(mode={args.mode})")

    if args.dry_run:
        assert "torch" not in sys.modules, "dry-run must not import torch"
        if not _has_all_scores(rows):
            print("dry-run-ok (no ML imports; plan only — candidates lack teacher_score)")
            return 0
        scored = rows   # already carries teacher_score
    else:
        from boldt_embed.config_teacher import load_teacher_models_config
        tcfg = load_teacher_models_config(args.teacher_config)
        scored = RT.score_lists_with_teacher(rows, tcfg, mode=args.mode)

    annotated = RT.annotate_lists(
        scored, positive_threshold=args.positive_threshold, hard_neg_margin=args.hard_neg_margin,
        temperature=args.temperature, use_teacher_only_positives=args.use_teacher_only_positives)
    summary = RT.summarize(annotated, positive_threshold=args.positive_threshold)
    print(f"[summary] {json.dumps(summary, ensure_ascii=False)}")

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in annotated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pathlib.Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"[write] {len(annotated)} scored lists -> {args.output}; summary -> {args.summary}")
    if args.dry_run:
        print("dry-run-ok (no ML imports; annotated from precomputed teacher_score)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
