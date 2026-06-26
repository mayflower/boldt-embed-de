# AutoResearch Runbook v8 — driving the program

How to drive the stateful frontier program (architecture: `docs/autoresearch-controller-2026.md`).
Everything defaults to **dry-run**; real GPU/teacher work happens only when you add the explicit
`--real`/`--allow-*` flags. Goal: beat the same-size-peer (e5-base/LFM2.5) MTEB(deu) retrieval-core
aggregate without regressing any task.

## The loop you run
```bash
make autoresearch-controller-dry         # status + the next planned trial
# …run the planned command (dry-run first, then real with its flags)…
conda run -n boldtembed python scripts/ar_controller.py record --event-json <event.json>   # log the result
make autoresearch-report                 # Pareto frontier + promotable candidates
```
`ar_controller.py next --dry-run` always prints the next command to run; record each result so the
ladder advances.

## End-to-end dry-run (no GPU — the smoke flow)
```bash
conda run -n boldtembed python scripts/ar_controller.py status
conda run -n boldtembed python scripts/ar_controller.py plan --trial-type data_mix --dry-run
conda run -n boldtembed python scripts/ar_build_mixture.py --config configs/autoresearch/mixtures/v8_balanced.json --catalog configs/data_sources.json --out outputs/autoresearch/mixtures/v8_balanced --dry-run
conda run -n boldtembed python scripts/ar_refresh_hardnegatives.py --config configs/autoresearch/hardneg_refresh.json --out outputs/autoresearch/hardneg/v8 --dry-run
conda run -n boldtembed python scripts/ar_train_specialist.py --config configs/autoresearch/specialists/v8_specialists.json --source-id swim_ir_de_full --out-root outputs/v8/specialists --dry-run
conda run -n boldtembed python scripts/ar_merge_search.py --config configs/autoresearch/merge_search_v8.json --out outputs/merged/v8_merge_search --dry-run
conda run -n boldtembed python scripts/ar_distill_trial.py --config configs/autoresearch/distill/listwise_kl_v8.json --dry-run
conda run -n boldtembed python scripts/ar_mteb_trial.py --config configs/autoresearch/mteb_retrieval_core.json --model outputs/v8/diverse-causal/checkpoint --label v8-diverse-causal --dry-run
conda run -n boldtembed python scripts/ar_report.py --format markdown
```

## Going real (each step, deliberately)
- mixture: `ar_build_mixture.py … --no-dry-run` (writes the big train.jsonl, git-ignored).
- specialist: `ar_train_specialist.py … --real --allow-gpu --allow-checkpoints` (hands to `ar_loop`).
- merge: `ar_merge_search.py … --real --allow-merge`.
- distill: `ar_distill_trial.py … --real --allow-gpu --allow-checkpoints` (new teacher lists:
  `ar_prepare_listwise_distill.py … --real --allow-gpu --allow-teacher` — GPU-days, de-risk a slice first).
- eval+promote: `ar_mteb_trial.py … --real --allow-gpu` then `ar_promote.py --candidate <label>`.

## Promotion rules (unchanged, fail-closed)
A candidate is promotable only if `check_mteb_frontier_gate.py` (run by `ar_promote.py`) passes:
its 4-task aggregate ≥ the same-size-peer frontier aggregate, **no** per-task regression below the
`v6-1-baseline-512` floor − tol, the baseline summary is present, and leakage is clean. Promotion
still needs human review; never claim a number beyond `outputs/mteb/<label>/summary.json` (ADR-005).
Specialists for a merge must share the warm-start basin; re-baseline at @512 for fair comparison.
