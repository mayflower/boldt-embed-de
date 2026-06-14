# v3 real-domain-generalization runbook

One command reproduces the whole v3 pipeline:
`scripts/run_v3_real_domain_experiment.py`. It orchestrates the v3 scripts (it does not
re-implement them) and is **safe by default**.

```bash
python scripts/run_v3_real_domain_experiment.py \
  --config configs/experiments/v3_real_domain_generalization.json \
  --manifest configs/data_sources_v3.json \
  --work-dir outputs/v3-real-domain \
  --mode full --target-count 100000 --device cuda --run-id-prefix v3 \
  --train-causal --train-reranker --eval \
  --i-understand-this-runs-gpu
```

## Modes

- `dry-run` (default): validates the config + manifest, writes the planned `COMMANDS.md` /
  `STATUS.json` / `V3_RESULTS.{md,json}`. **No torch, no downloads, nothing executed.**
- `smoke`: runs the CPU stages, prints/skips GPU stages (use `--allow-ml-smoke` to run them).
- `full`: executes everything — **requires `--i-understand-this-runs-gpu`**.

## Stages (in order)

1. validate v3 config + source manifest (fail-fast at startup, fail-closed manifest)
2. `acquire_v3_sources` (materialize real local drops)
3. `build_leakage_index` (over the eval corpora)
4. `build_v3_candidates` (manifest-gated, quotas, PII, `--fail-on-unknown-license`, `--fail-on-domain-quota-miss`)
5. `run_full_leakage_scan` (`--drop-hits` → cleaned candidates + report)
6. `build_bm25_index` (full corpus)
7. `build_teacher_cache` (sharded, both 8B teachers)
8. `summarize_teacher_cache` (`--fail-on-unknown-license` / `--fail-on-disallowed-training-source`)
9. `calibrate_teacher_thresholds` (embedder vs reranker filtered sets)
10. **`domain_quality_gate`** (blocks web/wiki dominance + low real-domain acceptance)
11. `mine_hard_negatives` (full corpus, `--require-full-corpus`)
12. `train_causal` (`--require-leakage-report`)
13. `train_bi_mntp` — only with `--train-bi-mntp` (default OFF; v2 causal won)
14. `build_reranker_candidates_v3` (high-precision, source-balanced)
15. `train_reranker`
16. `eval_dense` × {GermanQuAD, DT-test, GerDaLIR}
17. `matryoshka_sweep`
18. `reranker_lift` × {GermanQuAD, DT-test} + **`reranker_promotion_gate`**
19. v3 results summary (`V3_RESULTS.{md,json}`)
20. **`release_gate`** (`validate_release_2026.py --require-v3-artifacts`)

## Safety — you cannot accidentally promote a bad run

- **Unknown licenses** can't reach training: the manifest is fail-closed, `build_v3_candidates`
  uses `--fail-on-unknown-license`, and `summarize_teacher_cache` / the release gate fail on any
  unknown-license row.
- **Missing leakage scan**: `build_v3_candidates` requires the leakage index; `train_causal`
  uses `--require-leakage-report` (refuses to train without a clean/cleaned report).
- **Capped mining**: `mine_hard_negatives` runs with `--require-full-corpus` (any subsample fails).
- **Reranker degradation**: the promotion gate fails on any GermanQuAD/DT-test delta `< 0` or any
  domain dropping `> 0.02`, or on low-precision positives.
- **Domain quality**: stage 10 fails if real domains are absent/teacher-rejected or the effective
  set is web/wiki-dominated.

A **gate** stage (domain-quality, reranker promotion, release) aborts the run by default. With
`--allow-research-failures` the run continues but the **verdict becomes `invalid_for_promotion`**
— so a research run is possible, but it can never be reported as promotable.

## Outputs

`outputs/v3-real-domain/`: `COMMANDS.md` (exact planned commands), `STATUS.json` (per-stage
status + verdict), `V3_RESULTS.{md,json}` (verdict + stage roll-up), plus every stage's own
artifacts (candidates, leakage report, teacher cache + calibration, domain-quality, eval, lift,
gate).

Verdicts: `planned` (dry-run) · `smoke-ok` · `promotable` (full, all gates green) ·
`invalid_for_promotion` (a gate failed under `--allow-research-failures`) · `failed`.
