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

## v3: high-precision labels, source-balanced lists, neutral-or-better gate

v2 cut the GermanQuAD degradation from −0.354 to −0.040 but still failed promotion. The residual
cause: the reranker trained on the **same noisy positives** as the embedder (pos teacher-median ≤
neg median). v3 fixes the inputs and hardens the gate.

**Inputs.** Positives come from the *calibrated* `qwen3_v3.filtered_reranker.jsonl` (stricter
teacher threshold, `docs/teacher-calibration.md`). Candidates come from multiple first stages —
BM25 (full-corpus index), student dense (causal v2/v3), e5/teacher dense, teacher-reranker-mined —
so the reranker doesn't overfit one candidate distribution. Build with
`scripts/build_reranker_candidates_v3.py` (`--bm25-results/--dense-results/--e5-results/
--teacher-reranker-results`); it reports the candidate-source distribution and `lists_with_min_sources`.
GermanQuAD-style lists are built only from **non-eval** sources — the builder never reads eval corpora.

**Labels (`reranker_modern.v3_label`).**
- **positive**: teacher reranker `>= positive_threshold` (default **4.0** — high precision).
- **negative**: teacher reranker `<= positive_threshold − neg_margin` (default 2.0) — a *clear* negative.
- **uncertain**: in between → `label=null`. Uncertain candidates feed **listwise soft targets
  only**, never hard BCE negatives (`candidate_lists_to_pointwise` skips them).

**Loss (mixed default).** Pointwise BCE on confident labels only; pairwise margin only where the
teacher margin is strong (`--pairwise-min-teacher-margin`, e.g. 2.0); listwise KL over the **full**
candidate list (including uncertain) toward the teacher softmax.

**Training summary** (`reranker_training_summary`, written to `reranker_training_summary.json`):
positive/negative teacher-score separation by domain, uncertain count, candidate-source
distribution, synthetic-vs-real share, and `high_precision_positives`.

**Gate (`check_reranker_promotion_gate.py`).** Fails if:
- GermanQuAD delta `< 0.0` **or** DT-test delta `< 0.0` (neutral-or-better — catches v2's −0.040);
- any evaluated domain drops by more than `--catastrophic-degradation` (default 0.02);
- the reranker trained on **low-precision positives** (`--training-summary` →
  `high_precision_positives=false`) without `--allow-low-precision-positives`.

A −0.001 GermanQuAD delta still **fails** — v2's small degradation can no longer pass.

## v4: listwise-primary RAG reranker (`train_rag_reranker_v4.py`)

v4 trains a German **RAG** reranker on teacher-scored candidate lists
(`scripts/score_rag_candidate_lists.py` → `rag_train_scored.jsonl`), with **listwise distillation
as the PRIMARY objective** so pointwise BCE cannot dominate on noisy labels.

**Objectives (`--loss mixed_listwise`, `plan_rag_reranker_loss`):**
- **Listwise KL** over the per-list teacher distribution — prefers the precomputed
  `teacher_softmax_target`, else softmax of `teacher_score`; runs first and carries weight 1.0.
- **Pairwise margin** — gold positive > hard negative, only on a strong teacher margin
  (`--pairwise-min-teacher-margin`, default 2.0).
- **Pointwise BCE** — **restricted to high-confidence gold** (label 1 AND
  `high_precision_positive`) + clear hard negatives (label 0). Uncertain candidates
  (teacher-only positives + too-close) and low-confidence golds are **excluded** — BCE never sees
  noisy labels (weight 0.2).
- **Optional MSE** to `teacher_score` (`--with-mse`).

**Balanced sampling:** `domain_balanced_list_sampler` caps lists per domain (and per list source)
deterministically so no domain/source dominates.

**Leakage:** never trains on GermanQuAD/DT-test labels or the WebFAQ held-out split — the input
is the train split, and `--eval-query-ids` hard-fails if any eval query_id appears in training.

**Reporting** (`rag_reranker_training_report`, written to `rag_reranker_training_report.json`):
loss-by-component plan, examples by domain/source, gold positives, hard negatives, uncertain,
teacher-only positives, teacher-score separation, + a run card.

```bash
python scripts/train_rag_reranker_v4.py \
  --candidate-lists outputs/v4-rag-reranker/teacher/rag_train_scored.jsonl \
  --output outputs/v4-rag-reranker/checkpoints/boldt-rag-reranker-v4 \
  --loss mixed_listwise --bf16 --gradient-checkpointing --epochs 1 --batch-size 8 \
  --run-id v4-rag-reranker
```

