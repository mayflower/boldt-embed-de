# v6 standalone RAW RAG reranker

Trains a standalone Boldt German RAG reranker, **evaluated RAW (no serving policy)**, on
**positive-present candidate lists only**. The reranker is a no-op on a positive the first stage
never retrieved — so this work is **gated on first-stage recall being fixed**.

- Core: `src/boldt_embed/reranker_modern.py` (`build_v6_reranker_dataset`,
  `partition_lists_by_positive_presence`, `v6_pairwise_present_only`, `v6_pointwise_present_only`,
  `build_v6_listwise_batches`, `train_v6_raw_reranker`, `v6_raw_reranker_run_card`)
- CLI: `scripts/train_v6_raw_rag_reranker.py` — Tests: `tests/test_train_v6_raw_rag_reranker.py`

## Loss (no policy objective)

- **Listwise KL — PRIMARY** (distil the Qwen3-Reranker-8B candidate distribution).
- **Pairwise margin** — high-precision gold vs clear negative, teacher-margin gated, **positive-present lists only**.
- **Pointwise BCE** — confident labels only (high-precision gold / clear negative).
- Uncertain candidates are **listwise-only** (never a hard BCE/pairwise label).
- Optional rank-preservation diagnostic is **disabled by default**. **No policy loss is ever an objective.**

## The false-negative guard (the core v6 correctness fix)

**Candidate lists where the positive is absent are excluded from BCE and pairwise** — their
"negatives" would be *false negatives* (we'd teach the model that the right answer is wrong). Such
lists are kept only as **diagnostics**. `partition_lists_by_positive_presence` splits the lists; the
run card records `lists_positive_absent_excluded` and `absent_excluded_from_bce_pairwise: true`. This
is enforced and unit-tested.

## Promotion (raw, never policy)

Promotion uses the **RAW reranker gate** — lift over FIXED candidate lists — never a policy-gated
serving workaround (`no_serving_policy_in_promotion: true` in the run card). This is also enforced by
`validate_release_2026.py` (`check_no_policy_gated_recommendation`, `check_reranker_raw_recommendation`).

## Status: trainer READY — training BLOCKED on the precondition (honest)

The task's precondition is **train only after candidate recall is fixed**. As of 2026-06-15 that
precondition (first-stage recall is fixed) is now **MET** (corrected — see below). The remaining gap
is purely **data plumbing**: the v6 candidate union lists have not been built/teacher-scored yet, so
the reranker has not been trained. One blocker remains, and one earlier "blocker" was a mistake:

1. **No v6 candidate lists exist (remaining blocker).** The CLI's input
   (`data/processed/v6/reranker_train_lists_teacher_scored.jsonl`) does not exist. A v6 reranker must
   train on lists built by the **v6 dense retriever** (union with BM25), teacher-scored by
   Qwen3-Reranker-8B — that data has not been produced (it needs an 8B-teacher scoring pass).
2. **First-stage recall IS measured and IS materially improved (CORRECTION).** An earlier version of
   this doc claimed "no real WebFAQ corpus on disk" — that was **wrong**. The real eval corpus is
   `outputs/v4-rag-reranker/eval/webfaq/` (1,381 docs, 1,576 queries, qrels). Measured over it
   (`outputs/v6-dense-rag/webfaq_real_recall_bm25_vs_dense.json`): BM25 Recall@100 **0.651** → dense
   Boldt-v6 Recall@100 **0.964** (**+0.313**; Recall@10 0.638 → 0.739); missing-positive rate
   **0.349 → 0.036**. **The precondition is met.**
3. **The v5 teacher-scored lists are injected-gold, not real recall.** A dry-run of this trainer on
   them (`outputs/v5-small-rag/teacher/rag_train_scored.jsonl`) reports `present=5660,
   absent_excluded=0, present_rate=1.0` — every positive present **only because it was injected**.
   Training on these would reproduce the failed v5 reranker, so v6 must train on freshly-built
   dense-v6 ∪ BM25 union lists (blocker #1), not these.

**Decision: training is UNBLOCKED on recall; build the v6 candidate union lists (then teacher-score)
before training.** The earlier "do not train — recall not fixed" conclusion was based on a false
"no corpus" premise and is retracted.

## What is actually needed before v6 reranker training

1. Obtain/stand up a **real retrieval corpus** for WebFAQ (and local RAG) and run
   `scripts/audit_first_stage_recall.py` with **real dense-v6 candidate lists** — confirm WebFAQ
   Recall@100 materially exceeds the BM25 baseline (0.65) over the *real* corpus.
2. Build **v6 candidate union lists** (BM25 ∪ dense-v6, top-100/200) where positives are genuinely
   retrieved — and **teacher-score** them with Qwen3-Reranker-8B (high-confidence / uncertain flags).
3. Then run this trainer on `data/processed/v6/reranker_train_lists_teacher_scored.jsonl`:

```bash
python scripts/train_v6_raw_rag_reranker.py \
  --candidate-lists data/processed/v6/reranker_train_lists_teacher_scored.jsonl \
  --output outputs/v6-reranker/checkpoints/boldt-rag-reranker-v6 \
  --loss listwise_kl+pairwise+pointwise_confident --bf16 --gradient-checkpointing \
  --run-id v6-raw-reranker
# --dry-run writes the data report + loss plan + run card with NO ML imports.
```

4. Evaluate **RAW** lift over fixed candidate lists (no serving policy); promote only if the raw gate
   passes. GerDaLIR stays diagnostic-only.

## Acceptance criteria

- ✅ The reranker would be **trained on valid (positive-present) candidate lists only** — and since no
  valid v6 lists exist yet, it is **not trained** (the gate held; no injected-gold or policy shortcut).
- ✅ **No serving policy is part of the promotion criterion** — promotion is RAW lift, enforced by the
  release gate.
