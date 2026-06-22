---
description: Build the AutoResearch preparation manifest from LOCAL data (+ a leakage report)
argument-hint: "--train PATH --eval-manifest PATH [--require-leakage-report PATH]"
allowed-tools: Bash(conda run *) Read
disable-model-invocation: true
---
# AutoResearch — prepare data manifest

Build the preparation manifest from LOCAL files only (no downloads). Use the arguments I passed:

```
$ARGUMENTS
```

Run:

```bash
conda run -n boldtembed python scripts/ar_prepare.py $ARGUMENTS --out outputs/autoresearch/prepared
```

If I passed no arguments, do NOT guess paths — instead show me `--help` and list exactly which
inputs you need (a train JSONL, an eval manifest, and ideally a leakage report).

After it runs, read `outputs/autoresearch/prepared/prepare_manifest.json` and report: train record
count, eval sets (required vs optional, any missing), the **leakage status**, and whether the
preparation is `promotable`. A preparation without a verified-clean leakage report is **not
promotable** — say so plainly. This manifest is what `/ar-trial real` passes to the recipe so the
leakage status flows into the gate.
