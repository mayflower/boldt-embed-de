# v5 — small German RAG retriever + reranker (plan)

**Status:** planning. Config: `configs/experiments/v5_small_rag.json`
(validated by `src/boldt_embed/v5_rag_config.py`). Supersedes the v4 reranker track as the active
product target. v1–v4 are kept historical/diagnostic.

## Why v5 (what v4 taught us)

The v4 full run (RTX A6000, 2026-06-14, `outputs/v4-rag-reranker/V4_RAG_RESULTS.md`) produced a
**strong in-domain FAQ reranker** but it **did not generalize**:

| eval set | first stage | + reranker | Δ nDCG@10 |
|---|--:|--:|--:|
| WebFAQ held-out | 0.5945 | 0.8852 | **+0.2907** |
| GermanQuAD | 0.9058 | 0.8347 | **−0.0711** |
| DT-test | 0.9774 | 0.9767 | −0.0007 |

The promotion gate **failed** and the reranker stays *Experimental / not recommended*. Two root
causes drive the v5 design:

1. **Single-style training.** v4 trained only on WebFAQ FAQ pairs (question → short answer). It
   over-fit that style and reordered GermanQuAD QA passages *worse*. Fix: train on **diverse
   German RAG question styles** (FAQ, QA-passage, web non-FAQ, long-doc chunks, German stress,
   local RAG).
2. **Near-ceiling evaluation.** GermanQuAD/DT-test first stages were already near-ceiling (recall
   ≈ 0.96–0.99, oracle nDCG@10 = 1.0). Reranking a near-perfect list can only churn it, so a tiny
   negative delta there is **noise, not failure**. v4's gate treated those sets as hard
   neutral-or-better signals, which is the wrong test. Fix: a **near-ceiling policy** — sets with
   oracle nDCG@10 ≥ 0.98 get a small do-not-regress tolerance (−0.005) and are **never a primary
   promotion signal**; promotion is driven by sets with real headroom (WebFAQ held-out, local
   RAG, a hard private web-QA set).

Legal/admin retrieval and **GerDaLIR remain diagnostic-only** — never a release blocker for this
track (this was the v3→v4 decision and it carries forward).

## Goal

A **small, deployable** German RAG retriever + reranker that measurably **improves fixed
candidate lists** across FAQ, QA-passage, web, long-doc, and local RAG — small enough to run on
modest hardware, distilled from the 8B Qwen teachers.

## Candidate models (bake-off, not yet chosen)

- **Dense retrievers:** `boldt-modern-causal-v3` (current best in-house causal),
  `Qwen/Qwen3-Embedding-0.6B`, `BAAI/bge-m3`, `intfloat/multilingual-e5-base`.
- **Rerankers:** `boldt-rag-reranker-v4` (the v4 student), `Qwen/Qwen3-Reranker-0.6B`.
- **Teachers (distillation/labels only):** `Qwen/Qwen3-Embedding-8B`, `Qwen/Qwen3-Reranker-8B`.

## Training domains (diverse, leakage-safe)

`faq_real`, `qa_passage_non_eval`, `web_nonfaq`, `long_doc_chunks`, `german_stress`, `local_rag`.

Public-benchmark eval sets may **never** appear as a training domain (enforced by the config
validator). `local_rag` is private and intentionally trains on a split disjoint from its eval
split.

## Eval sets

`webfaq_heldout`, `germanquad_do_not_train`, `dt_test_do_not_train`, `local_rag`,
`hard_webqa_private`. The `_do_not_train` markers are explicit; GermanQuAD/DT-test are
near-ceiling diagnostics, not the promotion driver.

## Promotion gate (`success_criteria`)

Promote only if **all** hold:

- WebFAQ held-out lift ≥ **+0.05** nDCG@10 (higher bar than v4's +0.03 — we want real quality);
- local RAG lift ≥ **+0.03** (if present);
- hard private web-QA lift ≥ **+0.03**;
- GermanQuAD ≥ **−0.005**, DT-test ≥ **−0.005** (near-ceiling do-not-regress tolerance, not a lift target);
- no non-ceiling eval catastrophic: delta ≥ **−0.02**;
- dense 256-dim Matryoshka retention ≥ **0.95** of full-dim quality (deployable small vectors).

## Hardness-aware candidate lists

v4 used BM25 top-20 only. v5 mixes first stages (in-house dense + e5 + bge-m3 + BM25) and mines
**hard** negatives/positives so the fixed candidate lists are non-trivial to reorder — measuring
reranker skill where it matters, not on near-perfect BM25 lists.

## Non-goals

- Not a legal/admin retriever; GerDaLIR is diagnostic-only.
- Not chasing public-benchmark leaderboard numbers — public sets stay eval-only.
- Not a large model — "small + deployable" is a first-class constraint (256-dim retention gate).
