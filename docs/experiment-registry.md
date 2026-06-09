# Experiment registry & run cards

Every teacher-cache build, training run, and evaluation writes a **run card** — a small JSON
provenance record — so any number under `outputs/` is traceable to the exact command, commit,
environment, inputs, and outputs that produced it. This is the machine-readable backbone of
the release gate (Prompt 12). Code: `src/boldt_embed/experiment_registry.py`,
`scripts/summarize_experiments.py`. Pure stdlib.

## Run-card schema

```json
{
  "run_id": "...", "run_type": "teacher_cache|train_embedder|train_reranker|eval",
  "command": "...", "commit": "...", "date": "ISO-8601",
  "hardware": "...", "gpu": "...", "python": "...",
  "torch": "...", "transformers": "...", "sentence_transformers": "...",
  "input_artifacts": ["..."], "output_artifacts": ["..."],
  "model": "...", "dataset": "...", "metrics": {...}, "notes": "..."
}
```

`validate_run_card` requires `run_id`, `run_type` (∈ the four types), `command`, `commit`,
`date`. Library versions are read from package **metadata** (`importlib.metadata`) — collecting
env info imports no ML, so it works in `--dry-run` and tests.

## Writing run cards

Every real-run script takes `--run-id` (optional; auto-derived from run_type + commit +
timestamp if omitted) and calls `experiment_registry.emit_run_card(...)` on success, writing
`outputs/run-cards/<run_id>.json`. Wired into:

| script | run_type |
|---|---|
| `build_teacher_cache.py` | `teacher_cache` |
| `train_modern_embedder.py`, `prepare_bidirectional_student.py` | `train_embedder` |
| `train_modern_reranker.py` | `train_reranker` |
| `eval_reranker_lift.py`, `eval_hybrid_retrieval.py`, `run_baseline_benchmarks.py` | `eval` |

Dry-runs return *before* emitting (no card for a plan-only run).

## Summarize

```bash
python scripts/summarize_experiments.py                      # all cards -> outputs/EXPERIMENTS.md
python scripts/summarize_experiments.py --run-type eval      # filter by type
python scripts/summarize_experiments.py --model boldt-modern-bi --dry-run   # print, don't write
```

`summarize_experiments.py` reads `outputs/run-cards/*.json`, filters by `--run-type` /
`--model` / `--dataset`, and writes a sorted Markdown index (`outputs/EXPERIMENTS.md`) with
key metrics, commit, and date per run. Invalid cards are counted and skipped.

## Why

The v1 numbers were hard to trust because the command/commit/data behind them weren't always
recorded together. Run cards make every number self-describing and let the release gate refuse
to ship if the provenance (baseline run, leakage check, eval) is missing.
