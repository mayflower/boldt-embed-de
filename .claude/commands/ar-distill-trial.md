---
description: Listwise-KL distillation trial — validate lists, plan train + MTEB eval (dry-run default)
argument-hint: "<distill-config> [--real --allow-gpu --allow-checkpoints]"
allowed-tools: Bash(conda run *) Bash(cat *) Bash(tail *) Read
disable-model-invocation: true
---
# AutoResearch Distill Trial

Listwise-KL distillation from the teacher's ranking (Prompt 09). Validates the base checkpoint + the
teacher-scored lists (fail-closed: ≥2 candidates, a positive, a teacher signal, no eval source),
then plans the `train_listwise_kl.py` command + the downstream MTEB eval. Default: dry-run.

```bash
conda run -n boldtembed python scripts/ar_distill_trial.py \
  --config ${1:-configs/autoresearch/distill/listwise_kl_v8.json} --dry-run
```
Real: add `--real --allow-gpu --allow-checkpoints`. To score NEW teacher lists first (GPU-days):
`conda run -n boldtembed python scripts/ar_prepare_listwise_distill.py --config <cfg> --dry-run`
(then `--real --allow-gpu --allow-teacher`).

Report: the list validation stats, the planned train command, the output checkpoint, and the MTEB
eval plan. listwise-KL is domain-shaped — judge by `/ar-promote`, not the per-batch loss.
