---
description: Train ONE domain specialist on a single catalogue source (from a shared warm-start, for merging)
argument-hint: "<source_id> [steps] [warm-start-ckpt]"
allowed-tools: Bash(conda run *) Bash(tail *) Bash(cat *) Read Edit
disable-model-invocation: true
---
# AutoResearch — train a domain specialist

Train a specialist on ONE catalogue source so it can later be MERGED (`/ar-merge`) with others.
Parse `$ARGUMENTS`: first = `source_id` from `configs/data_sources.json` (must be
`training_usable` + `scanned_clean`); second (optional) = steps (default 6000); third (optional) =
warm-start checkpoint dir.

For merging to work the specialists MUST share a basin: warm-start ALL specialists from the SAME
checkpoint (the Phase-1 balanced model), not from the raw base. Default warm-start =
`outputs/v8/diverse-causal/checkpoint` (override with the third arg).

1. Edit ONLY `configs/autoresearch/experiments/current.json`: set `data_mixture={<source_id>: 1.0}`,
   `runtime.materialize_mixture=true`, `runtime.mixture_total=<source rows>`,
   `runtime.train_base_model="<warm-start>"`, `training.max_steps=<steps>`. One-line rationale.
2. Run one real trial (trains on the materialized single-source corpus, leakage clean-by-construction):
   ```bash
   conda run -n boldtembed python scripts/ar_loop.py --real --allow-gpu --allow-checkpoints \
     --status keep --run-id spec-<source_id> --out-root outputs/v8/specialists \
     --baseline outputs/autoresearch/baseline/metrics.json
   ```
3. Confirm integrity (only `configs/autoresearch/experiments/*.json` may have changed):
   `conda run -n boldtembed python scripts/check_autoresearch_integrity.py --format json`.
4. Report the run verdict + the specialist checkpoint path
   (`outputs/v8/specialists/spec-<source_id>/checkpoint`). Repeat for each domain (e.g. swim_ir_de_full,
   mmarco_de, synthetic_legal_v2), then `/ar-merge` them. Tell me before launching many (each is GPU-time).
