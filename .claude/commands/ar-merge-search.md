---
description: Merge-search over complementary specialists (soup/SLERP/TIES/DARE), dry-run default
argument-hint: "<merge-config> [--real --allow-merge]"
allowed-tools: Bash(conda run *) Bash(cat *) Bash(tail *) Read
disable-model-invocation: true
---
# AutoResearch Merge Search

Reproducible merge-search over complementary specialist checkpoints (Prompt 08): mean,
weighted_mean, slerp_pairwise, task_vector_sum, ties, dare_linear, layerwise_weighted_mean.
Unsafe-for-the-format methods are reported `unsupported` (not silently mis-merged). Default: dry-run
lists the planned merge candidates (no torch, no checkpoints loaded).

```bash
conda run -n boldtembed python scripts/ar_merge_search.py \
  --config ${1:-configs/autoresearch/merge_search_v8.json} \
  --out outputs/merged/v8_merge_search --dry-run
```
Real merges: add `--real --allow-merge` (writes per merge: checkpoint + merge_manifest.json +
parent_manifest.json under git-ignored outputs/merged/). Then `/ar-mteb-trial` + `/ar-promote` each.

Report: the method-support table, the planned/produced candidates, and — for real merges — which
candidate keeps each parent's strong task (escapes the trade-off). No benchmark claim without a saved eval.
