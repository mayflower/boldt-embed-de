---
description: Build (or dry-run) a leakage-clean training mixture from the data-source catalogue
argument-hint: "<mixture-config> [--real]"
allowed-tools: Bash(conda run *) Bash(cat *) Bash(tail *) Read
disable-model-invocation: true
---
# AutoResearch Build Mixture

Compose a manifested, leakage-clean training corpus from `configs/data_sources.json` (Prompt 04).
Default: dry-run (writes plan + manifest + report, NOT the big train.jsonl). Fail-closed on unknown /
training_usable:false / non-scanned_clean sources.

```bash
conda run -n boldtembed python scripts/ar_build_mixture.py \
  --config ${1:-configs/autoresearch/mixtures/v8_balanced.json} \
  --catalog configs/data_sources.json \
  --out outputs/autoresearch/mixtures/<name> --dry-run
```
Real build: replace `--dry-run` with `--no-dry-run` (writes the full train.jsonl, git-ignored).

Report: the planned source budgets, domain/length mix, dedupe counts, leakage status, and the
manifest path. Then the next step is `/ar-refresh-hardnegs` or a dense/specialist trial on this mix.
