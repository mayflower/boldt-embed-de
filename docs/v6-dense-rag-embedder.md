# v6 dense RAG embedder (first-stage recall)

Trains the next Boldt-based German **dense RAG retriever**, targeting **first-stage recall**
(`Recall@50/100` + candidate-list coverage) — **not** reranker lift. The recall audit
(`docs/first-stage-recall-audit.md`) showed the bottleneck is retrieval: WebFAQ BM25 recall@10 ≈
0.65, so ~35% of positives never reach the candidate list and **no reranker can recover them**. A
better dense retriever puts those positives into the lists.

- Core: `src/boldt_embed/train_modern.py` (`build_v6_dense_dataset`, `domain_balanced_examples`,
  `v6_dense_run_card`) — CLI: `scripts/train_v6_dense_rag_embedder.py`
- Tests: `tests/test_train_v6_dense_rag_embedder.py`

## Model

- **Primary: Boldt causal, mean-pooling.** v6 continues from `boldt-modern-causal-v3`
  (a `Boldt/Boldt-DC-350M` German-retrieval derivative — the current best in-house causal retriever),
  so it keeps German retrieval competence instead of relearning it. Pass `--model
  Boldt/Boldt-DC-350M` to start from the raw base instead.
- bi+MNTP is an ablation only (`--bidirectional`); causal is the default from prior evidence.

## Training data (leakage-safe; fails closed)

Built into `data/processed/v6/` from existing leakage-filtered sources (no public-benchmark text):

| source | domain | pairs |
|---|---|--:|
| v5 WebFAQ real | `faq_real` | 2,000 |
| v5 QA-passage non-eval | `qa_passage_non_eval` | 2,500 |
| v5 long-doc chunks | `long_doc_chunks` | 800 |
| v5 German stress | `german_stress` | 1,200 |
| v3 DT-de-dpr (train) | `wiki_non_eval` | 8,000 |
| v3 ger_backtrans/web | `web` | 8,000 |

**Hard negatives:** 33,948 triplets with **Qwen3-Reranker-8B teacher-score margins**, mined from the
v5 teacher-scored training lists (`outputs/v5-small-rag/teacher/rag_train_scored.jsonl`) — top-6
hardest non-positives per query, `teacher_margin = positive_score − negative_score`.

**Never trained on:** GermanQuAD eval/test, DT-test eval split, WebFAQ held-out, or any public
benchmark. `build_v6_dense_dataset` runs `leakage_reason` on every row and **fails closed** (the run
aborts with a non-zero exit if any row references a public benchmark / eval split). Synthetic pairs
flagged `must_teacher_validate` train only when a teacher score clears the threshold.

## Objectives / loss stack

1. **CachedMultipleNegativesRankingLoss** (contrastive base).
2. **MatryoshkaLoss** over **[1024, 768, 512, 256, 128, 64]** (deployable small vectors).
3. **MarginMSELoss** from teacher-score margins on hard negatives — *prepared* (data + plan).
4. Optional **EmbedDistillLoss** from Qwen3-Embedding-8B vectors (`--distill-vectors`).
5. **NO_DUPLICATES** batch sampler.
6. **Domain-balanced batches with a FAQ cap** (`--faq-cap`, default 0.5): FAQ is stride-sampled to at
   most `faq_cap` of the kept set, then domains are round-robin interleaved so no batch is FAQ-heavy.

**Honest note on the executed loss.** The current `train_modern_embedder` runs
**CachedMultipleNegativesRankingLoss → MatryoshkaLoss** under the `SentenceTransformerTrainer` (with
explicit hard negatives when every example has one, otherwise in-batch negatives). **MarginMSELoss is
prepared (the 34k teacher-margin triplets exist and the loss plan lists it) but is not yet wired into
multi-loss training** — that wiring (a `DatasetDict` of pairs+triplets with MNRL+MarginMSE) is the
documented next step. The run card records `executed_loss_stack` and `margin_mse_wired: false` so the
report never overstates what trained.

## CLI

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
python scripts/train_v6_dense_rag_embedder.py \
  --train-pairs data/processed/v6/rag_pairs_teacher_validated.jsonl \
  --hard-negatives data/processed/v6/hardneg_teacher_scored.jsonl \
  --output outputs/v6-dense-rag/checkpoints/boldt-dense-rag-v6 \
  --model outputs/v3-real-domain/checkpoints/boldt-modern-causal-v3 \
  --max-steps 1200 --batch-size 64 --bf16 --gradient-checkpointing \
  --faq-cap 0.5 --run-id v6-dense-rag
# device 0 is the slow Tesla P40; the A6000 is device 1 — train on CUDA_VISIBLE_DEVICES=1.
# --dry-run writes the data report + loss plan + run card with NO ML imports.
```

Reports: `outputs/v6-dense-rag/v6-dense-rag_run_card.json` + `outputs/run-cards/v6-dense-rag.json`
(train data by domain/source/license, hard-negative source + teacher-margin distribution, loss
stack, Matryoshka dims, domain balance, executed-loss note, GPU result).

## Evaluation — the target is RECALL, not reranker lift

This trains a **candidate**. Promotion is judged by **first-stage recall**, measured directly under
the harness — **not** reranker lift:

1. Build candidate lists for WebFAQ/local-RAG/GermanQuAD/DT-test with this retriever.
2. Re-run `scripts/audit_first_stage_recall.py` and compare **Recall@50/100** + candidate-list
   coverage against the BM25-only baseline (WebFAQ recall@10 0.65 today).
3. A v6 retriever earns promotion only if it raises WebFAQ recall (positives stop being absent),
   while GermanQuAD/DT-test stay within the near-ceiling do-not-regress tolerance and 256-d Matryoshka
   retention ≥ 0.95.

That evaluation is the next task; this task delivers the trained candidate + run card.
