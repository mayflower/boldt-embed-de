---
description: Refresh BM25/dense/teacher hard negatives + listwise candidate lists (dry-run default)
argument-hint: "<hardneg-config> [--real --allow-gpu --allow-teacher]"
allowed-tools: Bash(conda run *) Bash(cat *) Bash(tail *) Read
disable-model-invocation: true
---
# AutoResearch Refresh Hard Negatives

Mine fresh hard negatives + listwise candidate lists from a mixture (Prompt 05). Default: dry-run.
BM25-only is a real stdlib path; dense/teacher pools fail-closed without precomputed-embedding
artifacts; teacher scores are only loaded, never fabricated, and Qwen3 never runs here.

```bash
conda run -n boldtembed python scripts/ar_refresh_hardnegatives.py \
  --config ${1:-configs/autoresearch/hardneg_refresh.json} \
  --out outputs/autoresearch/hardneg/<label> --dry-run
```
Real: add `--real` (BM25/dense from artifacts); `--allow-gpu --allow-teacher` only for teacher-scored pools.

Report: the resolved candidate pools, false-negative filter stats, kept/dropped counts, per-domain
balance, and the two output files (hardnegatives.jsonl + listwise_candidates.jsonl).
