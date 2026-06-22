---
description: Autonomously run MULTIPLE AutoResearch rounds in one go (tune → trial → score → repeat)
argument-hint: "[rounds] [dry|real]"
allowed-tools: Bash(conda run *) Bash(tail *) Bash(cat *) Read Edit
disable-model-invocation: true
---
# AutoResearch — autonomous multi-round loop

Run several iterations **back-to-back without me triggering each one**, in THIS turn. Parse
`$ARGUMENTS`: first token = number of rounds **N** (default 5); second token = mode `dry` (default)
or `real`.

Setup: skim `AUTORESEARCH.md`. The editable surface is **only**
`configs/autoresearch/experiments/current.json`. PRIMARY metric = `webfaq_recall@100`; the
GermanQuAD / DT-test guardrails (Δ ≥ −0.005) and 256-d retention (≥ 0.95) must not regress. Keep a
running table and remember the best result + the config that produced it.

For round k = 1..N:

1. If k > 1, make **ONE** small, sensible change to `current.json` within editable fields
   (`loss.*`, `training.*`, `data_mixture`, `matryoshka_dims`, `pooling`, `normalize_embeddings`),
   guided by the verdicts so far — hill-climb the primary metric, and revert a change that hurt it.
   State a one-line rationale. On k = 1, run the current config unchanged.
2. Run exactly one iteration:
   - **dry:**  `conda run -n boldtembed python scripts/ar_loop.py --dry-run --status keep`
   - **real:** `conda run -n boldtembed python scripts/ar_loop.py --real --allow-gpu --status keep --baseline outputs/autoresearch/baseline/metrics.json --prepared-manifest outputs/autoresearch/prepared/prepare_manifest.json`
3. Read the JSON verdict, then run the integrity check:
   `conda run -n boldtembed python scripts/check_autoresearch_integrity.py --format json`.
   If it flags anything **outside** `configs/autoresearch/experiments/*.json`, **revert your last
   edit and STOP**.
4. Append the round to your table.

**Stop early** if: a round is `promotable: true`; OR 3 consecutive rounds show no WebFAQ
improvement; OR the integrity check fails.

At the end, print: a compact table (round · change · webfaq_recall · score · promotable), the best
config found, and the recommended next step. Never claim a benchmark beyond what the verdicts
report — dry-run numbers are plumbing only, and a real promotion still needs human review.

Note: in `real` mode each round runs on the RTX A6000 (~30–60s) and consumes context; keep N modest
(3–5) and tell me before launching a long real sweep.
