#!/usr/bin/env python3
"""Summarize run cards under outputs/run-cards/ into outputs/EXPERIMENTS.md (Prompt 11).

Pure stdlib. Filter by run_type / model / dataset. The table gives a one-glance index of
every traceable teacher-cache / training / eval run.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import experiment_registry as ER  # noqa: E402


def _matches(card, run_type, model, dataset):
    if run_type and card.get("run_type") != run_type:
        return False
    if model and model not in str(card.get("model") or ""):
        return False
    if dataset and dataset not in str(card.get("dataset") or ""):
        return False
    return True


def render_markdown(cards):
    lines = ["# Experiments", "", f"{len(cards)} run card(s).", "",
             "| run_id | type | model | dataset | key metrics | commit | date |",
             "|---|---|---|---|---|---|---|"]
    for c in cards:
        metrics = c.get("metrics") or {}
        keys = [k for k in ("ndcg@10", "mrr@10", "recall@100", "final_loss") if k in metrics]
        msummary = ", ".join(f"{k}={metrics[k]}" for k in keys) or "—"
        lines.append(f"| {c.get('run_id')} | {c.get('run_type')} | {c.get('model') or '—'} "
                     f"| {c.get('dataset') or '—'} | {msummary} | `{str(c.get('commit'))[:8]}` "
                     f"| {c.get('date')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-cards-dir", default=str(ER.RUN_CARD_DIR))
    ap.add_argument("--output", default=str(ROOT / "outputs" / "EXPERIMENTS.md"))
    ap.add_argument("--run-type", default=None, choices=sorted(ER.RUN_TYPES) + [None])
    ap.add_argument("--model", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cards = ER.read_run_cards(args.run_cards_dir)
    invalid = [c.get("run_id", "?") for c in cards if ER.validate_run_card(c)]
    cards = [c for c in cards if _matches(c, args.run_type, args.model, args.dataset)]
    cards.sort(key=lambda c: str(c.get("date")))
    print(f"[summarize] {len(cards)} matching run card(s); {len(invalid)} invalid")
    md = render_markdown(cards)

    if args.dry_run:
        print(md)
        return 0
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
