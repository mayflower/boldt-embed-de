---
description: Propose+apply ONE editable-surface config change toward WebFAQ recall, then run an iteration
argument-hint: "[hypothesis, e.g. 'lower loss.temperature to 0.02']"
allowed-tools: Bash(conda run *) Bash(tail *) Bash(cat *) Read Edit
disable-model-invocation: true
---
# AutoResearch — tune one knob, then iterate

Objective: improve the PRIMARY metric **webfaq_recall@100** without regressing the GermanQuAD /
DT-test guardrails (Δ ≥ −0.005) or 256-d Matryoshka retention (≥ 0.95).

Recent results:

!`tail -n 15 outputs/autoresearch/results.tsv 2>/dev/null || echo "(none yet — run /ar-trial first)"`

Current experiment config:

!`cat configs/autoresearch/experiments/current.json 2>/dev/null || echo "(missing)"`

Do this, then stop:

1. Considering my hypothesis (`$ARGUMENTS`, if any) and the results above, make **ONE** small,
   sensible change to `configs/autoresearch/experiments/current.json`. Edit **only** these fields:
   `loss.*`, `training.*`, `data_mixture`, `matryoshka_dims`, `pooling`, `normalize_embeddings`.
   **Never** edit a protected surface, the base config, or the recipe scoring/gates. State the
   one-line rationale.
2. Run one iteration (dry-run unless I explicitly said "real"):
   ```bash
   conda run -n boldtembed python scripts/ar_loop.py --dry-run --status keep
   ```
3. Confirm only the editable surface changed:
   ```bash
   conda run -n boldtembed python scripts/check_autoresearch_integrity.py --format json
   ```
4. Report: the change you made, the verdict, and the integrity result. If the integrity check
   flags anything **outside** `configs/autoresearch/experiments/*.json`, **revert your edit** and
   tell me.
