# Dense-recall STOP gate

A reranker cannot rank a document the first stage never retrieved. This gate refuses to waste GPU on
reranker training/eval when first-stage recall is insufficient (positives absent from candidate
lists). Pure stdlib.

- Gate: `scripts/check_dense_recall_gate.py` (`dense_recall_gate(metrics, targets)`)
- Enforced by: `scripts/train_v6_raw_rag_reranker.py` (refuses to train when the STOP file is active)
- Tests: `tests/test_dense_recall_gate.py`

## Inputs

Metrics are extracted defensively from any subset of:
- `--recall-report` — dense-vs-BM25 recall (`outputs/v6-dense-rag/webfaq_real_recall_bm25_vs_dense.json`)
- `--union-report` — candidate-union report (`outputs/v6-reranker/eval-lists/webfaq_union_report.json`)
- `--audit-report` — first-stage audit (`outputs/v6-dense-rag/first_stage_audit_webfaq.json`)
- `--local-rag-report` — optional local-RAG recall

## Gate conditions (pass only if all)

| metric | default target | source |
|---|---|---|
| WebFAQ Recall@100 | ≥ 0.95 | dense recall@100 (real corpus) |
| WebFAQ positive_in_top_50 | ≥ 0.90 | dense recall@50 |
| WebFAQ oracle_ndcg@10 | ≥ 0.95 | candidate-union present_rate (a present positive can be placed at rank 0) |
| missing_positive_rate | ≤ 0.10 | 1 − present_rate |
| candidate_union_size | ≥ 20 | union list_size |
| local RAG Recall@100 | ≥ 0.90 (if present) | local-RAG recall@100 |

Targets are tunable via `--target-recall-100 / --target-top-50 / --target-oracle`.

## Two-tier outcome (blocking vs advisory)

The acceptance condition is: **don't train when positives are ABSENT.** So the gate distinguishes:

- **`positives_absent = True`** — a recall-sufficiency check failed (Recall@100, missing-rate, oracle,
  or local-RAG recall). The reranker genuinely cannot recover the positives. → **BLOCKING:** writes
  `STOP_RERANKER_TRAINING.md` with *"Reranker cannot recover missing positives. Improve dense
  retrieval or candidate generation first."* The trainer refuses to run.
- **`positives_absent = False`** but a ranking-quality target (e.g. top-50) is below threshold →
  **ADVISORY:** the gate returns fail (targets not fully met) **but writes no STOP file** — positives
  are present, so reranker training is allowed (it simply won't promote unless it clears the raw
  reranker gate). Blocking training here would be wrong.

`check_dense_recall_gate.py` removes a stale STOP file whenever positives are present.

## Trainer enforcement

`train_v6_raw_rag_reranker.py` checks for `STOP_RERANKER_TRAINING.md` at the repo root **before any
GPU work**:
- present + no override → **refuses** (exit 2) with a pointer to fix dense retrieval first;
- `--force-research-run` → trains anyway, but the run card is stamped **`invalid_for_promotion:
  true`** and `forced_research_run: true` — a forced run can never be promoted.

## Current verdict (2026-06-16)

Run over the real WebFAQ corpus (`outputs/v6-dense-rag/dense_recall_gate.json`):

| check | value | target | status |
|---|--:|--:|---|
| Recall@100 | 0.964 | 0.95 | ✅ |
| positive_in_top_50 | 0.883 | 0.90 | ❌ |
| oracle_ndcg@10 | 0.966 | 0.95 | ✅ |
| missing_positive_rate | 0.034 | ≤0.10 | ✅ |
| candidate_union_size | 200 | ≥20 | ✅ |

**`positives_absent = False` → ADVISORY, not blocking.** Recall is sufficient (Recall@100 0.964,
present 0.966), so no STOP file is written and reranker training is permitted. The only miss is the
strict top-50 *ranking* target (0.883 vs 0.90) — and our reranker reranks the full 200-candidate union
(present 0.966), so present_rate/Recall@100 is the binding metric for it, not top-50. (Separately: the
v6 raw reranker already trained and **failed its own promotion gate for over-reranking near-ceiling
lists** — a model problem, not a recall problem; see `outputs/v6-reranker/raw_gate.md`.)

## Acceptance

- ✅ The pipeline cannot waste GPU on reranker training when positives are **absent** — the trainer
  refuses while a STOP file is active.
- ✅ Forced research runs are permitted but marked `invalid_for_promotion`.
