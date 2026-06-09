# Teacher score cache

The 2026 distillation workflow scores German (query, document) candidate pairs with strong
**teacher** models and caches the scores to JSONL. The student is later trained to match
those scores (distillation) and to rank teacher-confirmed positives above teacher-filtered
negatives. This document covers the cache format and how to produce it.

Teachers are **configurable** in `configs/teacher_models.json` — nothing is hard-coded to
Qwen. Defaults are `Qwen/Qwen3-Embedding-8B` and `Qwen/Qwen3-Reranker-8B` (both Apache-2.0,
32k context, instruction-aware), which fit one at a time on a 48 GB GPU.

## Input: candidate schema

`scripts/build_teacher_cache.py` reads a JSONL file of candidate pairs. Required fields:

| field | type | meaning |
|---|---|---|
| `query_id` | str | stable query id |
| `doc_id` | str | stable document id |
| `query` | str | German query text |
| `document` | str | German passage text |

Optional: `label` (number/null), `source` (str), `domain` (str), `positive` (bool/null).
See `tests/fixtures/teacher_candidates.jsonl` for a tiny example. Candidates are produced by
the dataset builder (`scripts/build_training_candidates.py`, Prompt 3).

## Output: cache schema (`teacher-cache-v1`)

One JSON object per scored pair:

```json
{
  "query_id": "q1", "doc_id": "d1",
  "query": "...", "document": "...",
  "label": null, "source": "synthetic", "domain": "admin", "positive": true,
  "embedding_teacher_model": "Qwen/Qwen3-Embedding-8B", "embedding_score": 0.78,
  "reranker_teacher_model": "Qwen/Qwen3-Reranker-8B",  "reranker_score": 3.91,
  "score_version": "teacher-cache-v1", "created_at": "2026-06-09T12:00:00+00:00"
}
```

- **`embedding_score`** — cosine similarity between the teacher query and document
  embeddings (both L2-normalized).
- **`reranker_score`** — the cross-encoder relevance score. **Raw** by default
  (`score_activation: "raw"` in the teacher config): a real-valued logit, *not* a
  probability. Set `score_activation: "sigmoid"` to store probabilities in `[0, 1]`. Raw
  scores are preferred for distillation (MarginMSE/KL) because they preserve margins;
  sigmoid is convenient for thresholding/inspection.

## Running it

```bash
# Dry-run: validate input schema + preview 3 planned rows. No torch, no GPU, no download.
python scripts/build_teacher_cache.py \
  --input tests/fixtures/teacher_candidates.jsonl --mode both --dry-run

# Real scoring (needs `.[train]` extras + GPU). One teacher loads at a time.
python scripts/build_teacher_cache.py \
  --teacher-config configs/teacher_models.json \
  --input data/processed/candidates.jsonl \
  --output outputs/teacher-cache/teacher_scores.jsonl \
  --mode both --device cuda --batch-size-embedding 8 --batch-size-reranker 4

# Resume an interrupted run: already-scored (query_id, doc_id) pairs are skipped.
python scripts/build_teacher_cache.py --input ... --output ... --resume
```

## 48 GB GPU recommendations

- Score **one teacher at a time**. The 8B teachers in bf16 are ~16–18 GB of weights plus
  activations; running embedding and reranker simultaneously risks OOM. `--mode both` loads
  them sequentially, not concurrently.
- Start with a small `--limit` (e.g. 100) and watch `nvidia-smi` before raising batch sizes.
- `flash-attn` is optional (`pip install flash-attn --no-build-isolation`); the loader falls
  back to eager attention if it is unavailable, so it is never required.
- Long German passages: the teachers support up to 32k tokens, but `max_length` in the
  config (default 8192) bounds memory — lower it for short FAQ/admin passages.

## Do not commit caches

Generated caches can be many GB. `outputs/teacher-cache/` is git-ignored. Only small JSON/MD
**reports** under `outputs/` are tracked. Never commit `teacher_scores.jsonl`.
