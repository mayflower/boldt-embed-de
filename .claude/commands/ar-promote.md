---
description: Run the MTEB(deu) frontier promotion gate for a candidate and write an auditable verdict
argument-hint: "<candidate-label>"
allowed-tools: Bash(conda run *) Bash(cat *) Bash(tail *) Read
disable-model-invocation: true
---
# AutoResearch Promote

Run the (protected) frontier gate on a candidate and record the verdict (Prompt 10). Does NOT
re-implement or weaken the gate — it invokes `check_mteb_frontier_gate.py` and saves
`promotion_verdict.json` + `promotion_report.md`. Fail-closed: a missing candidate/peer/baseline
summary fails.

```bash
conda run -n boldtembed python scripts/ar_promote.py --candidate ${1} --format markdown
```

Report: candidate aggregate vs the same-size-peer (e5-base/lfm2.5) frontier aggregate, per-task
deltas vs the @512 baseline, failed gates, and the `promotable` boolean. Promotion still needs human
review; no number is claimed beyond `outputs/mteb/<label>/summary.json`.
