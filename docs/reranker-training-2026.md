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

## v2: candidate-list training data (fix the GermanQuAD degradation)

The v1 reranker degraded GermanQuAD because it trained on one candidate distribution. v2 trains
on **candidate lists** built from multiple sources. `scripts/build_reranker_candidates_v2.py`
(via `negative_mining_2026.build_reranker_candidate_lists`) emits, per query, the positive
(label 1) + teacher-filtered hard negatives (label 0) drawn from BM25 + dense (student/e5/
teacher) sources, each tagged with `teacher_score`, `candidate_source`, `domain`:

```json
{"query_id": "...", "query": "...",
 "candidates": [{"doc_id": "...", "document": "...", "label": 1, "teacher_score": 6.9,
                 "candidate_source": "positive", "domain": "admin"},
                {"doc_id": "...", "document": "...", "label": 0, "teacher_score": -5.0,
                 "candidate_source": "bm25", "domain": "web"}],
 "positive_doc_ids": ["..."], "source": "...", "domain": "..."}
```

The report shows positives/negatives, vetoed false negatives, candidate counts by
source/domain, and pos-vs-neg teacher-score medians — so distribution mismatch is visible
before training.

### v2 mixed-loss training + promotion gate

`train_modern_reranker.py --candidate-lists reranker_train_v2.jsonl --loss mixed` trains on the
candidate lists with **pointwise (BCE) + pairwise (margin) + listwise (KL over teacher scores)**
combined (`reranker_modern.candidate_lists_to_{pointwise,pairwise,listwise}`).

**Anti-degradation gate** (`scripts/check_reranker_promotion_gate.py`): a reranker may NOT be
promoted unless it is **neutral-or-positive on every held-out set**. It reads the per-dataset
lift reports and fails if DT-test Δ < 0 **or GermanQuAD Δ < 0** (the v1 failure mode), with
+0.02 on GermanQuAD as the target:

```bash
python scripts/eval_reranker_lift.py --candidates eval/germanquad_shortlist.jsonl \
  --reranker outputs/checkpoints/boldt-reranker-modern-v2 --reranker-v1 outputs/checkpoints/boldt-reranker-modern \
  --output outputs/real-training/reranker-lift-germanquad-v2.json
python scripts/check_reranker_promotion_gate.py \
  --dt-test outputs/real-training/reranker-lift-dt_test-v2.json \
  --germanquad outputs/real-training/reranker-lift-germanquad-v2.json \
  --output outputs/real-training/reranker_gate.json   # exit 1 if it degrades either set
```

A model card may only call the reranker "recommended" once this gate passes (enforced by the
v2 release gate, Prompt 12).
