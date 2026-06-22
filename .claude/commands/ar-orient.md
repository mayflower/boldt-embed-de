---
description: Orient on the AutoResearch loop — goal, rules, editable vs protected surfaces, next step
argument-hint: ""
allowed-tools: Bash(conda run *) Read
disable-model-invocation: true
---
# AutoResearch — orientation

Read `AUTORESEARCH.md` in the repo root, then brief me in a few bullets on:

- the **goal** (improve German dense first-stage retrieval; WebFAQ recall@100 is the PRIMARY metric;
  GermanQuAD / DT-test are do-not-regress **guardrails**, never the primary signal),
- the **20-minute** per-trial budget,
- what I may edit — **only** `configs/autoresearch/experiments/current.json` and
  `src/boldt_embed/autoresearch_recipe.py` — versus the protected surfaces (scoring, gates, eval
  data, leakage checks, baselines, the base config),
- the fail-closed hard rules (train ≠ eval; leakage must be VERIFIED clean; dry-run numbers are
  plumbing only; 256-d retention ≥ 0.95).

Current protected-surface integrity status:

!`conda run -n boldtembed python scripts/check_autoresearch_integrity.py --format json 2>/dev/null || echo '{"status":"unknown"}'`

Then tell me the single best next action (usually `/ar-status`, then `/ar-trial dry`, then
`/ar-tune`). Do not change anything.
