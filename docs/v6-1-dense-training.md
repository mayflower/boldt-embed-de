# v6.1 dense training (WebFAQ Recall@50)

Trains **dense-v6.1** to lift WebFAQ **Recall@50** (dense-v6: 0.883 < 0.90) while preserving
Recall@100 (0.964), nDCG, the GermanQuAD/DT-test guardrails, and 256-d Matryoshka retention.
**DENSE-ONLY — no reranker is trained.**

- Core: `src/boldt_embed/train_modern.py` (`build_v6_1_dense_dataset`, `plan_v6_1_loss_stack`,
  `v6_1_dense_run_card`, `train_v6_1_dense_embedder`)
- CLI: `scripts/train_v6_1_dense_top50.py` — Tests: `tests/test_train_v6_1_dense.py`
- Config: `configs/experiments/v6_1_dense_top50.json` (`reranker_training_enabled: false`)

## Inputs

- Base checkpoint: `outputs/v6-dense-rag/checkpoints/boldt-dense-rag-v6` (continue training).
- Contrastive pairs: `data/processed/v6/rag_pairs_teacher_validated.jsonl` (22,500 multi-domain).
- Rank-promotion hard negatives: `data/processed/v6_1/dense_top50_hardnegatives.jsonl` — 1,430
  WebFAQ-train queries whose positive sits at dense rank 51–200, with the docs that outrank it
  (28,600 top-50 blockers, teacher-vetted by the mining veto).

## Losses

1. **CachedMultipleNegativesRankingLoss** — contrastive base.
2. **MatryoshkaLoss** over **[1024, 768, 512, 256, 128, 64]** (deployable small vectors preserved).
3. **MarginMSELoss (teacher margins)** — *data-level*: the rank-promotion negatives are teacher-vetted
   (the mining veto dropped any blocker within `veto_margin` of the positive's teacher score). It is
   **not** wired as a separate weighted loss in this run (`margin_mse_wired_as_separate_loss: false`).
4. **Rank-promotion loss** — realized as **CMNRL over (query, positive, top50-blocker) triplets**: the
   loss maximizes `sim(q, positive)` relative to the explicit blocker negatives, pushing the positive
   above the docs that currently keep it below rank 50. Triplets are built **only** for positives at
   dense rank 51–200, using **teacher-vetted** negatives.
5. **NO_DUPLICATES** sampler.

Training uses a `DatasetDict` of `{pairs, triplets}` sharing the `MatryoshkaLoss(CMNRL)` objective —
CMNRL adapts to the 2-column (pairs) and 3-column (triplets) batches, so both contrastive learning
and rank-promotion happen in one run.

## CLI

```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=/bigdata/johann/hf-cache HF_HUB_OFFLINE=1 \
python scripts/train_v6_1_dense_top50.py \
  --base-model outputs/v6-dense-rag/checkpoints/boldt-dense-rag-v6 \
  --train-pairs data/processed/v6/rag_pairs_teacher_validated.jsonl \
  --hard-negatives data/processed/v6_1/dense_top50_hardnegatives.jsonl \
  --output outputs/v6-1-dense-top50/checkpoints/boldt-dense-rag-v6-1 \
  --max-steps 1000 --batch-size 64 --bf16 --run-id v6-1-dense-top50
# device 0 is the A6000 (fastest-first ordering). --dry-run writes the report with NO ML imports.
# --gradient-checkpointing is supported but omitted in the executed run — memory is ample on the
# 49GB A6000 for a 350M model, and it ~halves throughput.
```

Run card (`outputs/run-cards/v6-1-dense-top50.json` + report under `outputs/v6-1-dense-top50/`):
rank-promotion triplets used + rank-51-100/101-200 counts, loss components, domain mix, hard-negative
teacher-margin distribution, Matryoshka dims, `reranker_trained: false`.

## Evaluation (dense-only; NEXT step, not this task)

Re-run `scripts/audit_first_stage_recall.py` + the BM25-vs-dense recall measurement over the real
WebFAQ corpus, the guardrails, and `scripts/check_dense_recall_gate.py` — v6.1 promotes only if
**Recall@50 ≥ 0.90** while Recall@100 / nDCG / guardrails / 256-d retention hold. Reranker work stays
deferred until dense-v6.1 is evaluated.

## Acceptance

- ✅ Produces a v6.1 dense checkpoint (`outputs/v6-1-dense-top50/checkpoints/boldt-dense-rag-v6-1`).
- ✅ No reranker checkpoint is trained (config `reranker_training_enabled: false`, asserted in the CLI;
  `reranker_trained: false` in the run card).
