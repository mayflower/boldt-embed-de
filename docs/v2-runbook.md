# v2 experiment runbook

`scripts/run_v2_generalization_experiment.py` runs the whole v2 pipeline reproducibly from one
command. It **orchestrates the existing scripts** (it does not reimplement them) and is **safe
by default**: `dry-run` validates + prints the plan, `smoke` runs the CPU-safe stages, and only
`full` (with `--i-understand-this-runs-gpu`) executes real GPU work. Every stage's underlying
script writes its own run card; the orchestrator writes `COMMANDS.md` + `STATUS.json` to the
work dir.

## Stages (in order)

build_candidates → generate_synthetic → teacher_cache (sharded) → summarize/filter cache →
mine_hard_negatives → reranker_candidate_lists → [train_causal] → [prepare_bi_mntp → train_bi_mntp]
→ [train_reranker] → [eval_dense ×3 + summarize_results].

The bracketed stages are gated by `--train-causal` / `--train-bi-mntp` / `--train-reranker` /
`--eval`.

## Modes

| mode | behavior |
|---|---|
| `dry-run` (default) | validate config+manifest (fails if `public_benchmarks_eval_only` is false), write `COMMANDS.md`/`STATUS.json`, print the plan. **No ML imports, nothing executed.** |
| `smoke` | run the stdlib/CPU stages for real (tiny `--target-count`); GPU stages are printed, not run (unless `--allow-ml-smoke`). |
| `full` | execute every stage via subprocess (fail-fast). **Requires `--i-understand-this-runs-gpu`.** |

## Commands

```bash
# Plan only (safe):
python scripts/run_v2_generalization_experiment.py --mode dry-run --target-count 1000 \
  --train-causal --train-bi-mntp --train-reranker --eval

# First smoke (CPU-safe stages real, GPU printed):
python scripts/run_v2_generalization_experiment.py --mode smoke --target-count 1000

# Full GPU run (explicit opt-in):
python scripts/run_v2_generalization_experiment.py --mode full --target-count 50000 \
  --teacher-mode both --train-causal --train-bi-mntp --train-reranker --eval \
  --device cuda --run-id-prefix v2 --i-understand-this-runs-gpu
```

All outputs go under `--work-dir` (default `outputs/v2-generalization/`) with v2 names.
`STATUS.json` records each stage's status (planned/ok/failed/skipped); the run stops fast on the
first failure. Generated checkpoints/caches under the work dir are git-ignored.
