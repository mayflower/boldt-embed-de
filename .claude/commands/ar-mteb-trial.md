---
description: Plan or run an MTEB(deu) retrieval-core eval for a candidate (dry-run default)
argument-hint: "<model-path> <label> [--real --allow-gpu]"
allowed-tools: Bash(conda run *) Bash(cat *) Bash(tail *) Read
disable-model-invocation: true
---
# AutoResearch MTEB Trial

Wrap `run_mteb_retrieval_de.py` reproducibly (Prompt 10): the fair same-size-peer comparison is at
@512; an optional native-context long-doc pass is planned when enabled in the config. Default:
dry-run prints the exact eval command(s) + writes a trial manifest.

```bash
conda run -n boldtembed python scripts/ar_mteb_trial.py \
  --config configs/autoresearch/mteb_retrieval_core.json \
  --model ${1} --label ${2} --dry-run
```
Real: add `--real --allow-gpu` (runs on the A6000; writes `outputs/mteb/<label>/summary.json`).

Report: the planned command(s), the summary path, and the next step (`/ar-promote <label>`). Never
claim a number beyond the saved summary (ADR-005).
