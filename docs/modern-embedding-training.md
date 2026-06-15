# Modern embedding training (2026)

A distillation-ready SentenceTransformers training path for the Boldt student, built on the
teacher cache. The legacy `train.py` (plain InfoNCE/MNRL/BCE) stays as a baseline; this is an
additive, modern path (`src/boldt_embed/train_modern.py`, `scripts/train_modern_embedder.py`).

## Loss stack

- **Base contrastive:** `CachedMultipleNegativesRankingLoss` — caches embeddings so the
  effective contrastive batch (and thus the number of in-batch negatives) can be far larger
  than fits in memory at once. With `--guide-model`, switches to `CachedGISTEmbedLoss`, which
  uses a guide model to filter false in-batch negatives.
- **Matryoshka:** the base loss is wrapped in `MatryoshkaLoss` over dims
  `[1024, 768, 512, 256, 128, 64]`, so a single model serves every truncation (re-normalize
  after truncating).
- **Distillation:** when the teacher cache carries scores, `MarginMSELoss` distills the
  teacher's (positive − negative) margin into the student.
- **Batch sampling:** `NO_DUPLICATES`, so a batch never contains the same text twice (which
  would create spurious in-batch negatives).

`plan_loss_stack()` reports exactly which losses will be built — and runs with no ML imports,
so `--dry-run` shows the plan offline.

## Input

Training examples are built from the teacher cache by grouping rows per query into
`{query, positive, negatives[], pos_score, neg_scores[]}`:

- **pair rows** → (query, positive) contrastive pairs,
- **rows with negatives** → (query, positive, hardest-negative) triplets,
- **teacher scores** → enable MarginMSE distillation.

Queries with no positive are skipped (a query the teacher never matched is unusable).

## Single RTX 6000 (48 GB) defaults & knobs

| knob | default | note |
|---|---|---|
| `--batch-size` | 32 | logical contrastive batch |
| `--mini-batch-size` | 8 | cached-loss forward chunk (lower if OOM) |
| `--bf16` | on (flag) | bf16 autocast |
| `--gradient-checkpointing` | on (flag) | trade compute for memory |
| `--lora` | off | PEFT adapter instead of full fine-tune |
| `max_seq_length` | 512 | raise for long passages, watch VRAM |

Start with a **10k-row first run** (`--max-steps 300`) to confirm loss goes down and VRAM is
in budget, then scale to the **full run** (`--epochs 1` over the whole cache). Watch
`nvidia-smi`; if you OOM, drop `--mini-batch-size` first, then `--batch-size`.

## Run it

```bash
# Offline plan (no torch): validate config, dataset metadata, loss stack
python scripts/train_modern_embedder.py \
  --teacher-cache outputs/teacher-cache/teacher_scores.jsonl --dry-run

# First (smoke-scale) GPU run
python scripts/train_modern_embedder.py \
  --teacher-cache outputs/teacher-cache/teacher_scores.jsonl \
  --output outputs/checkpoints/boldt-modern-smoke \
  --max-steps 300 --batch-size 32 --mini-batch-size 8 --bf16 --gradient-checkpointing

# Full run with a GIST guide model
python scripts/train_modern_embedder.py \
  --teacher-cache outputs/teacher-cache/teacher_scores.jsonl \
  --output outputs/checkpoints/boldt-modern-bi \
  --guide-model intfloat/multilingual-e5-base --epochs 1 --bf16 --gradient-checkpointing
```

## Resume & export

- **Resume:** the SentenceTransformerTrainer writes checkpoints under `--output`; restart
  pointing at the same dir (HF Trainer resumes from the latest checkpoint).
- **Export:** `export_sentence_transformers_model(model, dir)` (also done automatically at
  the end of training) saves a ready-to-load SentenceTransformer.

## Architecture guard

The student is loaded as an explicit `Transformer + Pooling` SentenceTransformer. If the base
cannot be loaded that way, `load_student_sentence_transformer` raises a clear error pointing
to the **bidirectional adapter** (`docs/bidirectional-adaptation.md`, Prompt 5) rather than
silently training the wrong architecture. Checkpoints are **never committed**
(`outputs/checkpoints/` is git-ignored).

## v2 training flags

For the v2 causal-vs-bi+MNTP comparison the trainer accepts:

- `--hard-negatives data/processed/hardneg_v2.jsonl` — train on **triplets** (anchor, positive,
  mined hard negative) from the miner, instead of the cache's (anchor, positive) pairs. A
  negative column is emitted only when **every** example has a negative (no positive-as-negative
  placeholders).
- `--base-model outputs/checkpoints/boldt-bi-mntp-v2` `--bidirectional true` — train the
  bidirectional student on its MNTP-adapted checkpoint (the eval re-applies the patch).
- `--use-teacher-score-distillation true|false|auto` — force MarginMSE teacher-score
  distillation on/off (auto = on when teacher scores are present and the config requests it).
- `--effective-batch-size` (informational), `--max-steps`, bf16 + gradient checkpointing for the
  48 GB profile.

```bash
# causal v2
python scripts/train_modern_embedder.py --teacher-cache .../qwen3_v2.filtered.jsonl \
  --hard-negatives data/processed/hardneg_v2.jsonl \
  --output outputs/checkpoints/boldt-modern-causal-v2 --bf16 --gradient-checkpointing
# bi+MNTP v2
python scripts/train_modern_embedder.py --base-model outputs/checkpoints/boldt-bi-mntp-v2 \
  --bidirectional true --teacher-cache .../qwen3_v2.filtered.jsonl \
  --output outputs/checkpoints/boldt-modern-bi-mntp-v2 --bf16 --gradient-checkpointing
```

## v5 dense RAG retriever (`scripts/train_v5_dense_rag_embedder.py`)

v5 trains a better small German **dense retriever** — not just another cross-encoder — so first-
stage recall improves and the candidate lists handed to the reranker are better. It composes the
stdlib helpers in `train_modern.py`: `build_v5_dense_dataset`, `plan_v5_dense_loss_stack`,
`v5_dense_run_card`.

**Loss stack** (objectives): `CachedMultipleNegativesRankingLoss` → `MatryoshkaLoss` over
`[1024, 768, 512, 256, 128, 64]` + `MarginMSELoss` (from Qwen3-8B teacher scores on hard negatives)
+ optional `EmbedDistillLoss` (MSE to Qwen3-Embedding-8B vectors), with the **NO_DUPLICATES** batch
sampler.

**Inputs**: `--train-pairs` (`data/processed/v5/rag_pairs.teacher_validated.jsonl`), optional
`--hard-negatives` (WebFAQ2 triplets with teacher margins), optional `--teacher-scores`
(`outputs/v5-small-rag/teacher/rag_embedder_teacher_scores.jsonl`), optional `--distill-vectors`
(Qwen3-Embedding-8B vectors → enables EmbedDistill).

**Models**: Boldt causal v5 (default) or `Qwen/Qwen3-Embedding-0.6B` with `--lora`.

**Guards (fail closed)**: rows referencing a public-benchmark eval set (GermanQuAD/DT-test/GerDaLIR
or the **WebFAQ held-out** split) are rejected — WebFAQ training pairs (`faq_real`) and WebFAQ 2.0
hard negatives (`webfaq2`) are legitimate training sources and pass. Synthetic pairs flagged
`must_teacher_validate` train ONLY when a teacher score clears `--teacher-threshold` (provisional
rows are excluded, not silently trained).

**Reports / run card**: loss stack, domain mix, hard-negative margin distribution, teacher
validation, Matryoshka dims, written to `outputs/v5-small-rag/<run-id>_run_card.json` and
`outputs/run-cards/<run-id>.json`. `--dry-run` writes these with **no ML import**.

```bash
python scripts/train_v5_dense_rag_embedder.py \
  --config configs/experiments/v5_small_rag.json \
  --train-pairs data/processed/v5/rag_pairs.teacher_validated.jsonl \
  --hard-negatives data/processed/v5/webfaq2_hardnegatives_de.jsonl \
  --output outputs/v5-small-rag/checkpoints/boldt-dense-v5 \
  --bf16 --gradient-checkpointing --max-steps 2000 --run-id v5-dense-boldt
```

> EmbedDistill ML wiring (MSE to teacher vectors) is declared in the plan/run card and activated by
> `--distill-vectors`; the contrastive+Matryoshka+MarginMSE stack runs via `train_modern_embedder`.
Matryoshka-dim retrieval quality is evaluated post-training with `scripts/eval_matryoshka_sweep.py`.
