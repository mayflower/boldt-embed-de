# AutoResearch Implementation Plan 2026

> **Superseded in part.** This was the build plan for the prompt-pack (prompts 01–13). The
> **deterministic controller / state-machine layer (Prompt 03: `ar_controller.py`,
> `autoresearch_state.py`, `events.jsonl`, the fixed ladder) was removed** in favor of **one
> Claude-orchestrated loop, `/ar-run`** (`AUTORESEARCH.md` + `docs/autoresearch-runbook-v8.md`):
> Claude reads the state off disk, picks the next lever, runs+gates it, repeats — no state file, no
> fixed order. The trial **tools** built by the other prompts (mixture builder, hard-neg refresh,
> specialist trainer, merge search, distill, MTEB eval/promote, reporting, Pareto) all remain — they
> are the levers the one loop composes. The historical plan is kept below for provenance.

This plan tracked the build that turned the single dense-trial loop into a research orchestrator
(the `boldt-embed-de-claude-code-prompt-pack`, prompts 01–13).

## Non-negotiable rules (every PR)
- **Never weaken eval data, benchmark harnesses, or release gates.** `data/processed/eval/**`,
  `src/boldt_embed/eval_harness.py`, `scripts/ar_score.py`, `scripts/check_*gate*.py`,
  `scripts/validate_release_2026.py` stay as-is (the orchestrator *calls* them, never edits them).
- **No model weights, checkpoints, HF caches, large datasets, or secrets committed.** All run
  artifacts land under git-ignored `outputs/`.
- **Fail-closed**: a missing leakage status / baseline / MTEB summary is a FAIL, not a warning.
  Unknown / non-`training_usable` / non-`scanned_clean` sources are rejected, naming the id.
- **New code is stdlib + dry-run by default.** Torch/transformers are imported lazily inside the
  real path only. GPU/teacher cost is gated behind explicit `--real`/`--allow-gpu`/`--allow-teacher`/
  `--allow-checkpoints`/`--allow-merge` flags.
- **No benchmark claim without a saved `outputs/.../summary.json` or `run_card.json`** (ADR-005).
- Tests + docs ship with every PR.

## Current state (the loop that exists)
- `AUTORESEARCH.md` — operating manual; `scripts/ar_loop.py` — one trial→score→log→integrity step;
  `scripts/ar_run_trial.py`, `scripts/ar_score.py`, `scripts/ar_sweep.py`; the recipe
  `src/boldt_embed/autoresearch_recipe.py`; base config `configs/autoresearch/base_dense.json`;
  overlay `configs/autoresearch/experiments/current.json`; `.claude/commands/ar-*.md`.
- Data: `configs/data_sources.json` (catalogue, source of truth) + `_materialize_data_mixture()`
  recipe hook (catalogue → one clean JSONL, fail-closed).
- Frontier program already instrumented: `scripts/model_soup.py`, `scripts/slerp_merge.py`,
  `src/boldt_embed/merging.py`, `scripts/train_listwise_kl.py`,
  `scripts/check_mteb_frontier_gate.py`, `scripts/ar_frontier_status.py`,
  `scripts/run_mteb_retrieval_de.py`, and the `/ar-frontier|data|specialist|merge|distill|mteb`
  commands.
- Hard-neg / listwise primitives: `src/boldt_embed/negative_mining_2026.py`.

## Protected surfaces (the loop must never edit; this build only *adds callers*)
`scripts/check_autoresearch_integrity.py` enforces: eval datasets, leakage checks, benchmark
harnesses, `scripts/ar_score.py`, the gate scripts, release gates, baseline outputs, and the base
config `configs/autoresearch/base_*.json`. (Prompt 02 edits `base_dense.json` once as a deliberate
maintenance fix — a catalogue-hygiene correction, committed so it becomes the protected baseline.)

## Target components & PR order

| component | existing files | planned files | risk | tests |
|---|---|---|---|---|
| 01 orientation | — | `docs/autoresearch-implementation-plan-2026.md` | none (doc) | unittest baseline |
| 02 config/catalog hygiene | `base_dense.json`, `autoresearch_recipe.py`, `data_sources.json` | recipe validation + fixed base mixture | editing base config | `test_autoresearch_mixture_validation.py` |
| 03 state machine + controller | `ar_loop.py` | `autoresearch_state.py`, `ar_controller.py`, `search_space_v8.json` | scope creep | `test_autoresearch_state.py`, `test_ar_controller.py` |
| 04 mixture optimizer | `_materialize_data_mixture`, `train_modern.domain_balanced_examples` | `data_mixture_optimizer.py`, `ar_build_mixture.py`, `mixtures/v8_balanced.json` | huge outputs | `test_data_mixture_optimizer.py` |
| 05 hardneg refresh | `negative_mining_2026.py` | `ar_refresh_hardnegatives.py`, `hardneg_refresh.json` | teacher cost | `test_ar_refresh_hardnegatives.py` |
| 06 dense trial generalization | `autoresearch_recipe.py`, `train_v6_1_dense_top50.py`, `train_modern.py` | new plan/training fields (grad-accum, eff-batch, temp schedule) | GPU OOM / silent no-op | recipe plan tests |
| 07 specialists | `.claude/commands/ar-specialist.md` | `ar_train_specialist.py`, `specialists/v8_specialists.json` | unattended GPU | `test_ar_train_specialist.py` |
| 08 merge search | `model_soup.py`, `slerp_merge.py`, `merging.py` | `merge_methods.py`, `ar_merge_search.py`, `merge_search_v8.json` | wrong merge math | `test_merge_methods.py`, `test_ar_merge_search.py` |
| 09 listwise-KL distill | `train_listwise_kl.py` | `ar_prepare_listwise_distill.py`, `ar_distill_trial.py`, `distill/listwise_kl_v8.json` | fabricated teacher scores | `test_ar_prepare_listwise_distill.py`, `test_ar_distill_trial.py` |
| 10 MTEB promotion | `run_mteb_retrieval_de.py`, `check_mteb_frontier_gate.py` | `ar_mteb_trial.py`, `ar_promote.py`, `mteb_retrieval_core.json` | overclaim | `test_ar_promote.py` |
| 11 reporting / Pareto | `ar_frontier_status.py`, `results.tsv` | `pareto.py`, `ar_report.py` | missing-as-0 | `test_pareto.py`, `test_ar_report.py` |
| 12 e2e + docs + commands | `AUTORESEARCH.md`, `Makefile`, `.claude/commands/` | controller/runbook docs, make targets, 9 slash commands | drift | full `unittest` + `validate_repo` |
| 13 hybrid ceiling-breaker (optional) | reranker, ADRs | `docs/research/...`, `hybrid_track.json`, stub scripts | distraction from dense | stub CLI tests |

## Risks
- **Unattended GPU / teacher spend** — every real path is flag-gated and dry-run is the default;
  the controller only *plans*.
- **Silent capability gaps** — a config knob with no real effect must fail-closed or be documented,
  never silently ignored (Prompt 06).
- **Merge math correctness** — merge methods are pure-stdlib and unit-tested on small state dicts
  before any real checkpoint touches them; unsafe methods report `unsupported`, not a bad merge.
- **Missing artifacts read as zero** — reporting marks them `missing`; gates fail closed.
- **Catalogue drift** — `mmarco_de`/`mqa_de` appear in both a train-pairs group and the
  `online_fetchable_status` status list; `_load_catalogue()` reads only the train-pairs groups, so
  validation resolves the real entries.

## Test strategy
Every PR ships unit tests runnable by `python -m unittest discover -s tests` (stdlib, no GPU). New
libraries assert `torch not in sys.modules` at import. CLIs are exercised in `--dry-run`. Real GPU
runs are out-of-band and never part of CI. Baseline at session start: `make test` / `python -m
unittest discover -s tests` green.
