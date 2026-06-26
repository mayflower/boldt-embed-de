---
description: AutoResearch reporting + Pareto frontier across WebFAQ/MTEB/cost metrics (read-only)
argument-hint: "[--candidate LABEL] [--format markdown|json]"
allowed-tools: Bash(conda run *) Bash(cat *) Bash(tail *) Read
disable-model-invocation: true
---
# AutoResearch Report

Read-only Pareto/frontier analysis over the saved artifacts (Prompt 11): events.jsonl, results.tsv,
outputs/mteb/*/summary.json, run metrics, and merge/distill/specialist manifests. Missing artifacts
are marked `missing`, never treated as 0.

```bash
conda run -n boldtembed python scripts/ar_report.py --format ${2:-markdown}
```
For one candidate: add `--candidate <label>`. Writes `outputs/autoresearch/reports/frontier.{json,md}`
+ `leaderboard.tsv` (pass `--no-write` to suppress).

Report: best-by-task, promotable candidates (MTEB aggregate ≥ same-size-peer frontier), the Pareto
frontier, regressions, and missing artifacts. No benchmark claim without an artifact path.
