# v6.1 — dense retriever improvement plan (WebFAQ top-50 recall)

**Scope: DENSE-ONLY. No reranker training.** v6.1 improves the Boldt dense RAG embedder's WebFAQ
**Recall@50** while preserving Recall@100 and the general guardrails. The raw reranker failed its gate
(`outputs/v6-reranker/raw_gate.md`); **no reranker work happens until dense-v6.1 is trained and
evaluated.**

- Config: `configs/experiments/v6_1_dense_top50.json` (validated by `src/boldt_embed/v6_1_dense_config.py`)
- Tests: `tests/test_v6_1_dense_config.py`

## Why v6.1 (what v6 left on the table)

Dense-v6 is the real product win — over the real WebFAQ corpus (`docs/dense-recall-gate.md`):

| metric | dense-v6 | v6.1 target |
|---|--:|--:|
| Recall@100 | **0.964** | ≥ 0.96 (preserve) |
| Recall@50 | **0.883** | **≥ 0.90** (the lever) |
| missing-positive rate | 0.036 | ≤ 0.04 (preserve) |
| nDCG@10 | 0.671 | ≥ 0.67 (preserve) |

The only shortfall is **top-50 recall (0.883 < 0.90)** — the dense-recall gate's *advisory* miss
(positives are present at Recall@100 0.964; this is a ranking-quality gap, not absence). v6.1 closes
it by sharpening the retriever's mid-rank precision (ranks ~10–50) without losing Recall@100.

## Approach

Continue training from `outputs/v6-dense-rag/checkpoints/boldt-dense-rag-v6` with **harder negatives
that target the top-50 boundary** — the docs that currently sit at ranks 51–100 and crowd out the
positive. Hard-negative sources (`hard_negative_sources`):

- `bm25_near_miss` — lexically-similar non-answers (BM25 confusions).
- `dense_v6_rank_51_to_200` — the retriever's own near-misses just below the cutoff (the docs that
  must be demoted to lift the positive into the top-50).
- `teacher_reranker_false_positive` — Qwen3-Reranker-8B-confident non-golds (teacher-validated hard).
- `e5_dense_near_miss` — a second dense retriever's confusions (diversity).

Training mix (`training_mix`, sums to 1.0): `webfaq_real 0.45`, `web_nonfaq 0.20`, `wiki_non_eval
0.20`, `german_stress 0.10`, `local_rag 0.05` — WebFAQ-weighted (the target domain) but kept
multi-domain so the general guardrails do not regress. Public benchmarks stay **eval-only**.

## Target metrics (promotion gate for v6.1)

From `target_metrics`:
- `webfaq_recall_at_50_min: 0.90` — **the v6.1 objective**
- `webfaq_recall_at_100_min: 0.96`, `webfaq_missing_positive_rate_max: 0.04`,
  `webfaq_ndcg_at_10_min: 0.67` — preserve the v6 recall win
- `germanquad_ndcg_at_10_min: 0.88`, `dt_test_ndcg_at_10_min: 0.94` — guardrails do not regress
- `matryoshka_256_retention_min: 0.95` — deployable small vectors retained

## Evaluation protocol (dense-only)

1. Train v6.1 from the v6 checkpoint with the boundary-hard negatives above.
2. Re-run `scripts/audit_first_stage_recall.py` + the BM25-vs-dense recall measurement over the **real**
   WebFAQ corpus, and the GermanQuAD/DT-test guardrails.
3. Re-run `scripts/check_dense_recall_gate.py` — v6.1 promotes only if **Recall@50 ≥ 0.90** while
   Recall@100/nDCG/guardrails/Matryoshka hold.
4. **Only after** dense-v6.1 is evaluated do we revisit the reranker — the dense-recall and raw
   reranker gates remain the promotion authority; **policy/bounded/abstain remain diagnostic-only**.

## Non-goals (enforced)

- **No reranker training** — `reranker_training_enabled: false`; the config validator rejects any
  config that sets it true, and no reranker trainer is invoked by this experiment.
- No public-benchmark training (`public_benchmarks_eval_only: true`).
- GerDaLIR/legal stays diagnostic-only; active RAG evals are WebFAQ / local RAG / GermanQuAD / DT-test.

## Acceptance

- ✅ v6.1 scope is **dense-only** (recall@50 lever; reranker untouched).
- ✅ No reranker training is triggered by this config (validated + fail-closed).
