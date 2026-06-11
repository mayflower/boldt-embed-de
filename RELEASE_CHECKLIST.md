# Release Checklist

Gate every public release on this list (implements ADR-006). Do not publish weights until
all **blocking** items pass.

## Blocking — licensing & provenance (ADR-001, ADR-004)
- [ ] Base-weight license confirmed: `Boldt/Boldt-DC-350M` = apache-2.0 (verified 2026-05-28).
- [ ] Every training dataset has a recorded, compatible license (`data.check_licenses` clean).
- [ ] Benchmark-leakage check clean against the eval registry (`data.find_leakage`).
- [ ] True parameter count verified; published model name is accurate (350M vs 0.5B — ADR-001).
- [ ] Base `config.json` read; 1024-d output (or projection head) decision finalized (ADR-003).

## Blocking — evaluation honesty (ADR-005)
- [ ] Real MTEB run executed; results saved under `outputs/` with full run metadata
      (command, commit, model, dataset, split, metric, hardware, output_path).
- [ ] Baselines evaluated under the same harness before any comparative claim.
- [ ] Matryoshka dims reported; German stress cases reported separately.
- [ ] No tuning against public test labels; train-time validation used a private dev split.

## Blocking — artifacts
- [ ] Model card per variant complete (intended use, limitations, evaluation, license, repro).
- [ ] No overclaims (no long-context, no "best multilingual" from a 350M German model).
- [ ] SentenceTransformers-compatible export verified to load and encode.

## Blocking — 2026 teacher/student workflow (run `python scripts/validate_release_2026.py`)
- [ ] Teacher model config recorded (`configs/teacher_models.json`).
- [ ] Teacher cache run card recorded (`outputs/run-cards/`, `run_type=teacher_cache`).
- [ ] Training data licenses recorded per candidate (`source`/`domain`/`license`; ADR-004).
- [ ] Leakage check clean against the eval registry (`filter_leakage_against_eval_texts`).
- [ ] PII scan clean (`tests/test_pii_schema.py` / data PII policy).
- [ ] Baseline report exists for the configured baselines (`scripts/run_baseline_benchmarks.py`).
- [ ] Student-vs-teacher comparison exists (same harness, same fixtures).
- [ ] Matryoshka dimension sweep exists (`scripts/eval_hybrid_retrieval.py`).
- [ ] Reranker lift report exists over **fixed** candidate sets (`scripts/eval_reranker_lift.py`).
- [ ] Each model card has provenance/leakage/stress/Matryoshka(or lift) sections + a
      **non-legal-advice** warning; no banned overclaim phrases.
- [ ] No model weights/checkpoints committed; no teacher cache committed under `outputs/teacher-cache/`.
- [ ] Every reported number has a **run card** (`docs/experiment-registry.md`,
      `outputs/EXPERIMENTS.md` via `scripts/summarize_experiments.py`).

## Blocking — v2 data-scale generalization (run `validate_release_2026.py --require-v2-artifacts`)
- [ ] v2 source manifest exists + validates (`configs/data_sources_v2.json`); public benchmarks blocked from training.
- [ ] v2 candidate-building report exists (`candidates_v2.report.json`).
- [ ] v2 teacher-cache summary exists (`teacher-cache/qwen3_v2.summary.json`); PII + leakage reports clean.
- [ ] v2 dense retrieval reports exist for GermanQuAD, DT-test, GerDaLIR.
- [ ] v2 causal-vs-bi+MNTP comparison exists (`summarize_v2_results.py` → V2_RESULTS).
- [ ] v2 Matryoshka sweep exists.
- [ ] v2 reranker lift report exists; **reranker promotion gate passes** (no GermanQuAD/DT-test
      degradation) before any model card calls the reranker "recommended".
- [ ] Public benchmark datasets are eval-only in BOTH `configs/data_sources_v2.json` and
      `benchmarks/mteb_german_tasks.json` (enforced by the gate's eval-leakage check).
- [ ] Every reported v2 number has a run card.

## Non-blocking — hygiene
- [ ] `make all` green; CI green on py3.10–3.12.
- [ ] Working tree clean; release tagged.
- [ ] `AUDIT.md` completed and signed off.

## Current status (updated 2026-06-11)
2026 teacher→student workflow **executed and measured** (v1): causal/bi+MNTP/reranker with
held-out numbers + run cards (`docs/benchmark-report.md` §6e–§6g). v2 data-scale-generalization
**infrastructure** is complete and validated (configs, manifest, builders, sharded teacher cache,
candidate lists, training/orchestration, dashboard, leakage-safe eval, this gate); the v2 **run**
is not executed yet. **Not release-ready** until the blocking items above are green.
