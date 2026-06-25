---
description: Listwise-KL distillation FT from a base checkpoint (teacher ranking), then MTEB-eval + gate
argument-hint: "<base-ckpt> [existing|new:<source>] [steps]"
allowed-tools: Bash(conda run *) Bash(tail *) Bash(cat *) Read
disable-model-invocation: true
---
# AutoResearch — listwise-KL distillation

Fine-tune a base checkpoint to match the Qwen3-Reranker's RANKING over candidate lists (the lever
that sharpens ranking without the contrastive over-fit). Parse `$ARGUMENTS`: first = base checkpoint
dir; second = `existing` (default, the cached 7500 v6 lists — CHEAP, no teacher cost) or
`new:<source_id>` (score NEW domain-matched data with the teacher — EXPENSIVE, GPU-days for the full
set); third (optional) = steps (default 1500).

**existing** (cheap, do this first):
```bash
CUDA_VISIBLE_DEVICES=0 conda run -n boldtembed python scripts/train_listwise_kl.py \
  --base <base-ckpt> --lists data/processed/v6/reranker_train_lists_teacher_scored.jsonl \
  --output outputs/v8/distill/<name>/checkpoint --steps <steps> --batch-queries 4 --list-k 24 \
  --lr 5e-6 --tau 0.05 --contrastive-weight 0.0 --run-id distill-<name>
```
Judge by the EVAL, not the per-batch loss (it is noisy: B=4 over heterogeneous lists).

**new:<source>** (expensive — confirm with me first; de-risk with a 20-50k slice before the full set):
1. `scripts/build_v6_candidate_union.py` → BM25+dense candidate lists for the source's queries.
2. `scripts/score_rag_candidate_lists.py --teacher-config configs/teacher_models.json` → Qwen3-Reranker-8B
   scores + `teacher_softmax_target` (this is the GPU-days step; gate behind a small slice).
3. `scripts/train_listwise_kl.py` on the new scored lists.

Then eval + gate via `/ar-mteb outputs/v8/distill/<name>/checkpoint <name>`.

Report: the eval deltas vs the base on all four tasks (listwise-KL is DOMAIN-SHAPED — it lifts the
tasks matching the teacher-list domain and can cost others; the frontier gate catches regressions),
and the frontier verdict. For `new:`, report the small-slice marginal lift before scaling.
