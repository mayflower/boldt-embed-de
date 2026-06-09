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

## Non-blocking — hygiene
- [ ] `make all` green; CI green on py3.10–3.12.
- [ ] Working tree clean; release tagged.
- [ ] `AUDIT.md` completed and signed off.

## Current status (2026-05-28)
Scaffold complete and validated. **Blocking eval + licensing-of-data items are open** because
no training/benchmark run has been executed. Not releasable yet — by design.
