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
  "label": null, "positive": true,
  "source_id": "local_admin_v2", "source": "local_admin_v2", "domain": "admin",
  "license": "CC-BY-4.0", "license_origin": "manifest", "allowed_for_training": true,
  "pair_hash": "p1",
  "embedding_teacher_model": "Qwen/Qwen3-Embedding-8B", "embedding_score": 0.78,
  "reranker_teacher_model": "Qwen/Qwen3-Reranker-8B",  "reranker_score": 3.91,
  "score_version": "teacher-cache-v1", "created_at": "2026-06-09T12:00:00+00:00"
}
```

### Provenance fields (carried verbatim from the candidate)

`make_cache_row` copies these from the candidate so the cache **never loses provenance**
(the v2 bug: licenses were dropped here, so every summary row collapsed to `"unknown"`):

- **`source_id` / `source`** — the manifest `source_id` the row was admitted under.
- **`domain`** — training domain (web / wiki_non_eval / faq / admin / …).
- **`license`** — the concrete license. For synthetic *inherited* sources this is the seed
  passage's real license (e.g. `CC-BY-SA-4.0`), falling back to the `…-inherits-source` marker.
- **`license_url`** *(optional)* — present only when the manifest/row provides it.
- **`license_origin`** — `manifest` (license read from the source entry) or `inherited`
  (synthetic data inheriting its seed passage's license).
- **`inherited_from_source_id`** *(optional)* — for `inherited` rows, the seed source.
- **`allowed_for_training`** — the manifest's training-permission bit, carried per row.
- **`pair_hash`** — used for near-duplicate detection in the summary.

Provenance is derived once by `source_manifest.candidate_provenance(entry, row)` so the manifest
stays the single source of truth.

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

## v2: sharding, resume, quality report (50k–250k candidates)

For v2 scale, `build_teacher_cache.py` writes **shards** so scoring is restartable and
parallelizable:

```bash
# Sharded scoring (one 8B teacher loaded at a time; --max-length is the memory knob):
python scripts/build_teacher_cache.py \
  --input data/processed/candidates_v2.jsonl \
  --output outputs/teacher-cache/v2/qwen3_v2.jsonl \
  --mode both --shard-size 5000 --max-length 512 --resume
#   -> outputs/teacher-cache/v2/qwen3_v2.shard-00000.jsonl, .shard-00001.jsonl, ...
#   -> outputs/teacher-cache/v2/qwen3_v2.manifest.json
# Parallel/manual: add --shard-index N to score just one shard.
```

`--resume` skips already-scored (query_id, doc_id) rows **per shard**. Then check usability
before training:

```bash
python scripts/summarize_teacher_cache.py \
  --input outputs/teacher-cache/v2/qwen3_v2.manifest.json \
  --output outputs/teacher-cache/v2/qwen3_v2.summary.json \
  --filter-output outputs/teacher-cache/v2/qwen3_v2.filtered.jsonl \
  --review-output outputs/teacher-cache/v2/qwen3_v2.review.jsonl \
  --reranker-threshold 2.0
```

The summary reports rows by source/domain/license, embedding/reranker score distributions,
missing-score counts, **suspicious low-scoring positives** (a generated query the teacher can't
match to its passage = a bad pair), and near-duplicate counts. `--filter-output` keeps positives
with reranker score ≥ threshold (low-scoring positives go to the review file with a
`filtering_reason`); negatives are kept for hard-negative mining.

### License / training-permission gates (provenance)

The summary also reports, per cache:

- **`by_license`** and **`by_license_origin`** — license histograms (real licenses, not
  `"unknown"`).
- **`unknown_license_rows`** — rows whose license is missing/empty/`"unknown"`. **Must be 0**
  before training/release.
- **`disallowed_for_training_rows`** — rows whose source has `allowed_for_training=false`
  (e.g. a public benchmark that leaked into the candidate set). **Must be 0.**
- **`synthetic_inherits_source`** — count of inherited-license rows + a breakdown by the seed
  source they inherit from (`by_inherited_from_source_id`).

Turn these into hard failures:

```bash
python scripts/summarize_teacher_cache.py --input <cache> --output <summary> \
  --fail-on-unknown-license --fail-on-disallowed-training-source   # exit 1 if either > 0
```

The release gate enforces the same invariant: `validate_release_2026.py --require-v2-artifacts`
(or `--require-v3-artifacts`) **fails** if any teacher-cache summary under the results dir has
unknown-license rows > 0. (Historical aside: the original v2 cache summary reported
`by_license {"unknown": 44336}` because `make_cache_row` dropped the license — fixed; the v2
summary was re-derived license-clean with `scripts/backfill_teacher_cache_license.py`.)

## Do not commit caches

Generated caches can be many GB. `outputs/teacher-cache/` (incl. `v2/` shards, manifest,
summary, filtered) is git-ignored. Only small JSON/MD **reports** under `outputs/` are tracked.
Never commit `teacher_scores.jsonl` or the shard files.
