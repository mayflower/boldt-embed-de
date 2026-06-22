---
description: Show AutoResearch status — recent results, current experiment config, best-so-far
argument-hint: ""
allowed-tools: Bash(cat *) Bash(tail *) Read
disable-model-invocation: true
---
# AutoResearch — status

Current experiment config (`configs/autoresearch/experiments/current.json`):

!`cat configs/autoresearch/experiments/current.json 2>/dev/null || echo "(missing)"`

Recent iterations (tail of the audit log):

!`tail -n 15 outputs/autoresearch/results.tsv 2>/dev/null || echo "(no results.tsv yet — run /ar-trial)"`

From the above, summarize for me:
- how many iterations have run, split by `mode` (dry_run vs real),
- the best **real** `webfaq_recall100` so far and which `run_id` produced it,
- whether any run was **promotable**, and the most common `failed_gates`,
- what the current config is set to and one hypothesis worth trying next.

Reminder: only `real` rows with a verified-clean `leakage_status` and passing guardrails are
promotable; `dry_run` rows are plumbing only. Do not run or change anything else.
