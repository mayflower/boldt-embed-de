# v4 — German RAG reranker plan

**Status: active product target.** Config: `configs/experiments/v4_rag_reranker.json`
(validated by `boldt_embed.v4_rag_config`).

## Why v4 (what changed from v3)

v3 asked "can real domain data fix admin/FAQ/legal generalization?" and answered honestly:

- **Real FAQ works.** WebFAQ (real, CC-BY-4.0) validated at **70.8%** teacher acceptance
  (reranker ≥2.0) vs v2's **synthetic** FAQ at **5.7%** — the v2 failure was the synthetic
  queries, not the domain.
- **Dense causal v3** (same harness): GermanQuAD **0.885**, **DT-test 0.970 (best causal yet)**,
  GerDaLIR (legal) **0.089**.
- v3's verdict was `invalid_for_promotion` **only because the v3 config required real admin +
  legal domains**, which we never sourced.

The product goal is now a **good German RAG reranker** — not legal/admin transfer. So v4 drops
admin/legal as release blockers. **GerDaLIR (legal) is kept as a DIAGNOSTIC only**
(`legal_eval_is_diagnostic_only: true`); it never gates a v4 release.

## Goal

Train and gate a German **RAG reranker** that lifts fixed first-stage candidate sets — measured
as nDCG@10 delta of (first stage) → (first stage + reranker) — without degrading any held-out
distribution.

## Inputs

- **First stage / dense default:** the v3 causal student
  (`outputs/v3-real-domain/checkpoints/boldt-modern-causal-v3`).
- **Teacher:** `Qwen/Qwen3-Reranker-8B` (high-precision labels via the v3 calibration:
  positives ≥4.0, uncertain=null → listwise only).
- **Candidate sources (≥3, source-balanced):** `bm25`, `v3_dense`, `e5_dense`, `qwen_dense`,
  `webfaq_hard_negatives`.
- **Train domains (no eval sources):** `faq_real` (WebFAQ), `web`, `wiki_non_eval`,
  `german_stress`.

## Eval sets

- `webfaq_heldout` — held-out real FAQ (the primary RAG target).
- `germanquad`, `dt_test` — must not be degraded (neutral-or-better).
- `local_rag` — a real local German RAG shortlist set (to be assembled).
- `gerdalir` — **diagnostic only**, reported but never a gate.

## Success criteria (gates)

| metric | min |
|---|---|
| `webfaq_reranker_delta_ndcg10_min` | **+0.03** (must genuinely lift real FAQ) |
| `germanquad_reranker_delta_ndcg10_min` | 0.0 (neutral-or-better) |
| `dt_test_reranker_delta_ndcg10_min` | 0.0 |
| `local_rag_reranker_delta_ndcg10_min` | **+0.03** |
| `catastrophic_degradation_max` | −0.02 (no domain may drop more) |

These reuse `check_reranker_promotion_gate.py` (neutral-or-better + catastrophic + high-precision
positives), now anchored on RAG/FAQ lift rather than legal transfer.

## Plan (reuses the v3 toolchain)

1. Build fixed first-stage shortlists per eval set (bm25 + v3_dense + e5 + qwen + WebFAQ hard negs).
2. Teacher-score + **calibrate** (embedder ≥2 / reranker ≥4) — `calibrate_teacher_thresholds.py`.
3. Build **high-precision, source-balanced** reranker candidate lists — `build_reranker_candidates_v3.py`.
4. Train mixed-loss reranker — `train_modern_reranker.py` (`--pairwise-min-teacher-margin`).
5. Eval lift on every eval set — `eval_reranker_lift.py`; **promotion gate** on the criteria above.
6. Legal (GerDaLIR) reported as a diagnostic line only.

## Out of scope (explicit)

- No admin/legal domain targets, and no admin/legal release blockers (legal is diagnostic).
- No synthetic queries over Wikipedia for real domains (v2 lesson).
