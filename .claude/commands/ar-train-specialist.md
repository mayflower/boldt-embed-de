---
description: Train ONE domain specialist from a shared warm-start (dry-run plans the ar_loop command)
argument-hint: "<source-id> [specialists-config] [--real --allow-gpu --allow-checkpoints]"
allowed-tools: Bash(conda run *) Bash(cat *) Bash(tail *) Read
disable-model-invocation: true
---
# AutoResearch Train Specialist

Train one domain expert on a single catalogue source, warm-started from the SHARED basin so a later
merge works (Prompt 07). Default: dry-run (writes the experiment config + manifest + the ar_loop
command it would run). Fail-closed on non-usable/unscanned source or a missing local warm-start.

```bash
conda run -n boldtembed python scripts/ar_train_specialist.py \
  --config ${2:-configs/autoresearch/specialists/v8_specialists.json} \
  --source-id ${1} --out-root outputs/v8/specialists --dry-run
```
Real: add `--real --allow-gpu --allow-checkpoints` (hands off to `ar_loop.py`, which materializes the
single-source clean mix and trains to `outputs/v8/specialists/spec-<label>/checkpoint`).

Report: the resolved label, warm-start, steps, planned command, and checkpoint path. Train each
domain (swim_ir_de_full, mmarco_de, mqa_de) then `/ar-merge-search`. Tell me before launching many.
