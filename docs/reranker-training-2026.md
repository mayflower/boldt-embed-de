# Reranker training (2026)

A stronger reranker path with three objectives + teacher distillation, all fed from the
teacher cache / mined hard negatives. The legacy BCE reranker (`train_reranker_de.py`) stays
as a baseline. Code: `src/boldt_embed/reranker_modern.py`, `scripts/train_modern_reranker.py`,
`scripts/eval_reranker_lift.py`.

## Objectives

| `--loss` | loss | signal | when |
|---|---|---|---|
| `pointwise` | `BCEWithLogits` (or MSE) | per-pair label / teacher score | simple, robust default |
| `pairwise` | `MarginRankingLoss` | score(pos) > score(neg) + margin | direct ranking |
| `listwise` | `KLDivLoss` | student dist → teacher softmax over candidates | **distillation** |
| `mixed` | all of the above | — | combine signals |

- **Pointwise** binary labels come from `positive`; with `label_mode="teacher"` the (sigmoid)
  teacher score becomes a soft regression target.
- **Pairwise** crosses each positive with each mined hard negative (capped per query).
- **Listwise** turns each query's candidate teacher scores into a softmax target and distills
  the student's candidate distribution toward it — this is where the **Qwen3 reranker teacher**
  is transferred into the small student.

Model: `AutoModelForSequenceClassification` (num_labels=1) on `Boldt/Boldt-DC-350M`, bf16 +
gradient checkpointing by default, optional LoRA.

## Measure LIFT over FIXED candidate sets — not retrieval

The single most important rule (and the v1 lesson): **a reranker is only meaningful as lift
over a fixed first stage.** `eval_reranker_lift.py` takes pre-built shortlists
(`{query_id, query, candidates:[{doc_id, document}], positive_ids}`) and reports nDCG@10 for:

- **first-stage** — candidates in the order given,
- **+ student reranker** — re-sorted by the trained student,
- **+ teacher reranker** — re-sorted by the Qwen3 teacher (ceiling reference),
- **oracle** (positives first) and **positive-in-top-k** — headroom.

Because the candidate set is fixed, the number isolates the reranker; it is never confounded
with a retriever's recall. A no-op scorer reproduces the first-stage number exactly (tested),
so any delta is real reranker effect.

## Run it

```bash
# Offline: build examples for the objective and print counts (no torch)
python scripts/train_modern_reranker.py --teacher-cache outputs/teacher-cache/teacher_scores.jsonl \
  --loss listwise --dry-run

# Train (GPU) — listwise distillation from the teacher
python scripts/train_modern_reranker.py \
  --teacher-cache outputs/teacher-cache/teacher_scores.jsonl \
  --hard-negatives data/processed/hard_negatives.jsonl \
  --output outputs/checkpoints/boldt-reranker-modern \
  --loss listwise --epochs 2 --batch-size 16 --bf16

# Evaluate lift over fixed shortlists (student + teacher ceiling)
python scripts/eval_reranker_lift.py \
  --candidates data/processed/eval_shortlists.jsonl \
  --reranker outputs/checkpoints/boldt-reranker-modern \
  --teacher-config configs/teacher_models.json
```

`--dry-run` on the eval still computes the **first-stage and oracle** numbers (pure stdlib),
so you can sanity-check shortlists before spending GPU time. Checkpoints are never committed.
