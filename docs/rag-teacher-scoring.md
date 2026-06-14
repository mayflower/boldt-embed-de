# RAG teacher scoring

Teacher-score **every** candidate in a RAG list — not only positive pairs — so the reranker can
distill the teacher's **listwise** distribution and train on **high-precision** labels. Module:
`boldt_embed.rag_teacher_scoring` · CLI: `scripts/score_rag_candidate_lists.py`.

## Inputs / outputs

- in: `data/processed/v4/rag_reranker_train_lists.jsonl` (+ eval lists for analysis),
  `configs/teacher_models.json`.
- out: `outputs/v4-rag-reranker/teacher/rag_train_scored.jsonl`,
  `…/rag_eval_webfaq_scored.jsonl` (analysis only), `…/rag_teacher_summary.json`.

## What it adds per candidate

- **`teacher_score`** — Qwen3-Reranker-8B relevance (optionally `embedding_score` cosine with
  `--mode both`). First-stage scores and `candidate_source` are preserved.
- **`teacher_rank`** — 1-based rank within the list by teacher score.
- **`teacher_softmax_target`** — softmax of the list's teacher scores (sums to 1); the listwise
  distillation target.
- **`high_precision_positive`** — gold positive whose teacher score ≥ threshold.
- **`uncertain`** — see policy.

## Labeling policy

- **Gold positives** (doc_id ∈ `positive_doc_ids`) stay positive (`label=1`).
- A non-gold candidate the teacher scores **high** (≥ `positive_threshold`, default 4.0) is a
  **teacher-only positive** → `uncertain=true`, **`label=null`** (listwise distillation only),
  unless `--use-teacher-only-positives` (then `label=1`).
- A candidate **too close** to the positive band (between `positive_threshold − margin` and
  `positive_threshold`) is `uncertain=true`, `label=null` — **never a hard BCE negative**.
- A **hard negative** must be clearly below (≤ `positive_threshold − margin`, default margin 2.0)
  → `label=0`.

So uncertain candidates feed the listwise KL target but are excluded from pointwise BCE — they
cannot be misused as hard negatives.

## Summary (`rag_teacher_summary.json`)

Teacher-score separation per domain (pos vs neg median), `uncertain_fraction`,
`candidate_source_quality` (median teacher + `pct_ge_threshold` per source), and
**teacher-vs-gold disagreements**: `gold_low_teacher` (a gold positive the teacher scores low)
and `teacher_only_positive` (a non-gold the teacher scores high). High disagreement = noisy gold
labels or a strong first stage surfacing unlabeled relevants.

## CLI

```bash
python scripts/score_rag_candidate_lists.py \
  --input data/processed/v4/rag_reranker_train_lists.jsonl \
  --teacher-config configs/teacher_models.json --mode reranker \
  --output  outputs/v4-rag-reranker/teacher/rag_train_scored.jsonl \
  --summary outputs/v4-rag-reranker/teacher/rag_teacher_summary.json
```

`--dry-run` never imports torch: it annotates from any `teacher_score` already on the candidates
(or just prints the plan) and writes the summary — for CI / wiring checks. Real scoring needs the
`train` extras + a GPU and loads one 8B teacher at a time.
