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

## v4 — German RAG reranker (active product track; run `validate_release_2026.py --require-v4-rag-artifacts`)
- [x] v4 config exists (`configs/experiments/v4_rag_reranker.json`).
- [x] WebFAQ held-out eval split exists; train/held-out are leakage-disjoint (deterministic hash split).
- [x] Fixed candidate lists exist (`candidate_lists/rag_reranker_train_lists.jsonl`).
- [x] Teacher-scored candidate lists exist (`teacher/rag_train_scored.jsonl`).
- [x] Reranker lift reports exist (`eval/reranker_lift_*.json`) over FIXED candidate sets.
- [x] Promotion gate report exists (`eval/rag_reranker_gate.json`).
- [x] Reranker model card claims "Recommended for German FAQ/RAG reranking" **only if** the
      promotion gate passes (WebFAQ/local lift ≥ +0.03, GermanQuAD/DT-test ≥ 0, no domain < −0.02).
- [x] Card always states: not legal advice, not a dense retriever, candidate-lists-only, lift-over-first-stage.
- [x] **No legal/admin corpus is required** for this track; GerDaLIR is diagnostic-only and never gates.

**Run executed 2026-06-14 (RTX A6000) — promotion gate FAILED → NOT promoted.** WebFAQ held-out
+0.2907 (pass), GermanQuAD −0.0711 (fail neutral + catastrophic), DT-test −0.0007 (fail neutral).
Card correctly stays *Experimental; not recommended for production reranking*. See
`outputs/v4-rag-reranker/V4_RAG_RESULTS.md`.

## Reranker promotion rule (applies to every reranker track)
- [ ] A reranker may be called "recommended" **only** when its **RAW** reranker shows lift over
      **FIXED** candidate lists and passes its raw promotion gate (`eval/v5_rag_lift_gate.json` /
      `eval/rag_reranker_gate.json` status `pass`). This is enforced by `validate_release_2026.py`
      (`check_reranker_raw_recommendation`).
- [ ] **Policy-gated variants do NOT count for model promotion.** Rerank-or-abstain, conservative +
      rank-preservation, preservation grids, and bounded `margin_override` serving policies are
      **diagnostics only** and must **never** be recommended as a production workaround
      (`check_no_policy_gated_recommendation` fails the gate if a card does). Policy docs are allowed
      only under diagnostics/analysis framing.

## v5 — small German RAG (EXECUTED; reranker NOT promoted; closed → v6)
- [x] Training data is leakage-filtered vs public guardrails (dt_test + GermanQuAD) and
      **demonstrably not FAQ-only** (FAQ share 0.217).
- [x] Reranker evaluated by the **hardness-aware gate** (`scripts/eval_v5_rag_lift.py`), not raw
      WebFAQ lift.
- [x] Reranker model card / README claim "recommended" **only if** the RAW reranker gate passes.
- [ ] **v5 RAW reranker promotion gate** passes (medium+hard lift on primary; guardrails
      do-not-regress; catastrophic rate ≤ 5%). — **FAILS.**

**Reranker run executed 2026-06-15 (RTX A6000) — RAW hardness-aware gate FAILED → NOT promoted.**
WebFAQ +0.1665 (primary pass), DT-test +0.0211 (pass), GermanQuAD −0.0285 with 16.9% catastrophic
drops (fail). The follow-on **policy experiments are diagnostic only**: the frozen bounded policy was
evaluated on a held-out near-ceiling guardrail and **also FAILED** its promotion gate (WebFAQ policy
Δ +0.0245 < +0.05). Failure analysis (`docs/v5-policy-failure-analysis.md`) shows the WebFAQ
under-lift is **mostly first-stage recall failure** (positives absent from candidate lists), which no
reranker can recover. Card/README stay *Experimental; not recommended* (raw **and** policy-gated).
See `outputs/v5-small-rag/V5_RESULTS.md`.

## v6 — dense RAG recall + standalone reranker (active product track; run `validate_release_2026.py --require-v6-dense-artifacts --require-v6-raw-reranker-artifacts`)
- [x] Plan recorded (`docs/v6-dense-rag-and-reranker-plan.md`).
- [x] **Dense first-stage recall** improved + measured directly under the harness over the real
      corpus: WebFAQ **Recall@100 0.651 → 0.964** (`docs/dense-recall-gate.md`).
- [x] **Standalone reranker** trained on multi-domain teacher-scored union lists and measured as
      **RAW** lift over FIXED candidate lists (no serving policy).
- [x] **GerDaLIR/legal is diagnostic-only** for the RAG track; active RAG evals are **WebFAQ / local
      RAG / GermanQuAD / DT-test**.

**Gate rules (enforced by `validate_release_2026.py`):**
- [ ] **Dense embedder** is recommended **only if the dense-recall gate passes** AND recall/eval
      reports exist — `check_v6_dense_recommendation`. (Currently advisory-fail on top-50; not promoted.)
- [ ] **Reranker** is recommended **only if the RAW reranker gate passes** —
      `check_v6_raw_reranker_recommendation`. (**Raw gate FAILED 2026-06-16:** WebFAQ Δ +0.036,
      GermanQuAD Δ −0.086 / 21% catastrophic → over-reranks near-ceiling; NOT promoted.)
- [x] **Policy-gated/bounded/abstain results are diagnostic-only** and may **never** be promotion
      evidence (`check_no_policy_result_as_promotion_evidence` + `check_no_policy_gated_recommendation`).
- [x] **No model card implies a serving wrapper is required** to make a model safe.
- [x] **No public-eval leakage** (`check_no_public_eval_leakage_v6` + manifest eval-only check).
- [x] The dense embedder may be recommended **independently** of the reranker.

**Run executed 2026-06-16 (RTX A6000).** Dense retriever = the win (recall fixed). Raw reranker =
NOT promoted (over-reranks near-ceiling guardrail lists; a model problem, not recall, and not maskable
by a serving policy). See `outputs/v6-reranker/raw_gate.md`, `outputs/v6-dense-rag/dense_recall_gate.json`.

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
