# WebFAQ / WebFAQ 2.0 hard negatives

Makes WebFAQ 2.0 hard negatives a **first-class v5 training source** instead of relying only on
negatives mined by our own pipeline. WebFAQ 2.0 publishes a hard-negative dataset (~1.25M queries
across 20 languages, up to 200 negatives/query with cross-encoder scores) usable for **MNRL** and
**MarginMSE**. We load it, filter it, and convert it for **both** the small dense embedder and the
reranker.

- Module: `src/boldt_embed/webfaq2_loader.py` (pure stdlib core, no network by default)
- CLI: `scripts/import_webfaq2_hardnegatives.py`
- Output: `data/processed/v5/webfaq2_hardnegatives_de.jsonl` (+ `…reranker_lists.jsonl`)

## Fail-closed policy

- **Local JSONL is the default path; no network in tests.** The Hugging Face loader is opt-in via
  `--download-hf` and lazily imports `datasets`; it is never touched on the local/dry-run path
  (asserted in the CLI).
- A **missing local file** (without `--download-hf`), an **unknown/absent license**, or a record
  **without a positive cross-encoder score** is rejected — the import fails rather than guessing.
- `--download-hf` **requires** `--hf-dataset` (no hardcoded id; confirm it against the WebFAQ 2.0
  release before use).

## Input record (flexible)

```
{
  "query": "Wie hoch darf die Mietkaution sein?",
  "positive": "Die Mietkaution darf hoechstens drei Nettokaltmieten betragen.",
  "positive_score": 8.0,
  "negatives": [
    {"document": "...", "cross_encoder_score": 5.5, "title": "...", "url": "..."}
  ],
  "language": "de",
  "license": "CC-BY-4.0",
  "title": "Mietkaution",
  "source_url": "https://example.de/mietkaution"
}
```

The normalizer also accepts `question`/`positive_document`/`answer` aliases, and negatives given
as a list of strings with a parallel `negative_scores` array.

## Preserved fields

query, positive answer/document, negatives, `cross_encoder_score` per candidate, language, source
URL/title (when present), and license — all carried through to the converted rows.

## Filtering

For each negative, `margin = positive_score - negative_score`:

- **keep** iff `margin >= --min-cross-encoder-margin` (default 2.0) — a clearly-worse-than-positive
  hard negative;
- **drop as false negative** if `margin <= --false-negative-margin` (default 0.5) — scored about
  as relevant as the positive, so likely actually relevant (counted as `dropped_false_negatives`);
- **drop as insufficient margin** otherwise (counted as `dropped_insufficient_margin`).

Kept negatives are sorted **hardest-first** (smallest qualifying margin) and capped to
`--max-negatives-per-query` (default 32) **deterministically** (tie-break by document hash).

## Two training outputs

1. **Embedder triplets** — one row per `(query, positive, negative)` with `teacher_margin`
   (= positive_score − negative_score) for **MarginMSE**; the same triples drive **MNRL** with
   in-batch + explicit hard negatives.
2. **Reranker candidate lists** — one row per query: the positive + kept negatives as candidates,
   each with its `teacher_score` and a listwise `teacher_softmax_target` (softmax over the
   cross-encoder scores). Shape matches the v4 RAG candidate-list / listwise convention.

## CLI

```
python scripts/import_webfaq2_hardnegatives.py \
  --input data/raw/webfaq2/de_hardnegatives.jsonl \
  --output data/processed/v5/webfaq2_hardnegatives_de.jsonl \
  --language de \
  --min-cross-encoder-margin 2.0 \
  --max-negatives-per-query 32 \
  --dry-run
```

`--dry-run` validates and writes the **report** but no data files, and imports no network/ML.
Reranker lists go to `--reranker-output` (default: `<output>.reranker_lists.jsonl`).

## Report

`imported_queries`, `negatives_per_query` (total/avg/min/max), `margin_distribution` (histogram
buckets `<0 … >=5`), `dropped_false_negatives`, `dropped_insufficient_margin`,
`capped_out_negatives`, `by_license`, `by_language`, `skipped_other_language`, `status`/`errors`.

## Acceptance

- WebFAQ hard negatives are a first-class training source (their own loader + report + tests).
- The same import feeds **both** the small dense embedder (margin triplets) and the reranker
  (listwise candidate lists).
