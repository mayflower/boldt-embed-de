# AutoResearch Controller (2026) — the stateful research orchestrator

The v8 program turns the single dense-trial loop into a **stateful state machine**: every planned /
run / observed trial appends one event to `outputs/autoresearch/events.jsonl`, and a controller
reads that log to decide the next trial. The controller is conservative — it **plans** (prints the
command) and never starts GPU/teacher work itself.

## Trial types & the deterministic ladder
`src/boldt_embed/autoresearch_state.py` defines the trial types and the `decide_next` ladder
(evaluated top-down over success-counts of the event log):

```
data_mix -> dense -> hardneg_refresh -> specialist (×2) -> merge -> distill -> mteb -> promotion
```

1. no clean mixture → `data_mix` · 2. no dense candidate → `dense` · 3. no refreshed
hardnegs/lists → `hardneg_refresh` · 4. < 2 specialists → `specialist` · 5. ≥ 2 specialists →
`merge` · 6. merge done → `distill` · 7. distill done → `mteb` · 8. mteb done → `promotion`.

## Components (each its own script + config + tests)
| trial | script | config | what it does |
|---|---|---|---|
| controller | `scripts/ar_controller.py` | `configs/autoresearch/search_space_v8.json` | status / next / plan / record |
| data_mix | `scripts/ar_build_mixture.py` | `configs/autoresearch/mixtures/v8_balanced.json` | catalogue → manifested clean corpus (`src/boldt_embed/data_mixture_optimizer.py`) |
| hardneg_refresh | `scripts/ar_refresh_hardnegatives.py` | `configs/autoresearch/hardneg_refresh.json` | BM25/dense/teacher negatives + listwise lists (`negative_mining_2026`) |
| dense | `scripts/ar_loop.py` | `configs/autoresearch/experiments/current.json` | a dense trial (generalized knobs: grad-accum, mini-batch, seq cap) |
| specialist | `scripts/ar_train_specialist.py` | `configs/autoresearch/specialists/v8_specialists.json` | one domain expert from the shared warm-start |
| merge | `scripts/ar_merge_search.py` | `configs/autoresearch/merge_search_v8.json` | soup/SLERP/TIES/DARE search (`src/boldt_embed/merge_methods.py`) |
| distill | `scripts/ar_distill_trial.py` (+ `ar_prepare_listwise_distill.py`) | `configs/autoresearch/distill/listwise_kl_v8.json` | listwise-KL from the teacher ranking |
| mteb | `scripts/ar_mteb_trial.py` | `configs/autoresearch/mteb_retrieval_core.json` | MTEB(deu) retrieval-core eval |
| promotion | `scripts/ar_promote.py` | — | runs the protected frontier gate, writes the verdict |
| reporting | `scripts/ar_report.py` | — | Pareto frontier across WebFAQ/MTEB/cost (`src/boldt_embed/pareto.py`) |

## Event schema (`events.jsonl`)
`{event_id, timestamp_utc, trial_type, status (planned|running|ok|fail|skipped), parent_artifacts,
input_artifacts, output_artifacts, config, metrics, gates, notes, git:{commit,dirty}}`. Build via
`autoresearch_state.new_event(...)`; append with `append_event`; record an external result with
`ar_controller.py record --event-json <file>`.

## Fail-closed invariants (preserved end-to-end)
- Mixtures: only `training_usable` + `scanned_clean`/`clean` catalogue sources (validated in
  `validate_recipe_config` when `materialize_mixture`, and in the mixture optimizer).
- Distill: lists must carry a teacher signal + a positive + ≥2 candidates; teacher scores are never
  fabricated; Qwen3 runs only behind `--allow-teacher`.
- Promotion: the protected `check_mteb_frontier_gate.py` is the only arbiter; missing
  candidate/peer/baseline summaries fail closed; no number is claimed beyond a saved summary.
- Generalized dense knobs that aren't truly active are listed in `plan_only_knobs` (never silently
  assumed active).
- No weights/large corpora committed (git-ignored); GPU/teacher only behind explicit flags.
