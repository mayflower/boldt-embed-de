# v5 RAG data mixer

Builds the v5 **domain-balanced, leakage-safe** German RAG training mix so v5 cannot silently
become another FAQ-only run (the v4 failure mode). Pure stdlib, no ML, no network.

- Module: `src/boldt_embed/v5_data_mixer.py`
- CLI: `scripts/build_v5_rag_data.py`
- Config: `configs/experiments/v5_small_rag.json` (validated by `src/boldt_embed/v5_rag_config.py`)

## Why

v4 trained on a single style (WebFAQ FAQ pairs) and lifted WebFAQ +0.29 nDCG@10 but degraded
GermanQuAD −0.07 — it did not generalize. v5 deliberately trains on **diverse** German RAG
question styles. The mixer makes that mechanical: a too-FAQ-heavy mixture is a **hard failure**,
not a silent default.

## Input schema (one JSON object per line)

```
{
  "source_id": "webfaq-000123",
  "domain": "faq_real|qa_passage_non_eval|web_nonfaq|long_doc_chunks|local_rag|german_stress",
  "query": "Wie hoch darf die Mietkaution sein?",
  "document": "Die Mietkaution darf hoechstens drei Nettokaltmieten betragen ...",
  "title": "Mietkaution",
  "answer": "hoechstens drei Nettokaltmieten",
  "license": "CC-BY-4.0",
  "source_url": "https://...",
  "synthetic_query": false,
  "generation_method": "teacher-qwen3-8b",
  "eval_only": false,
  "public_benchmark": false
}
```

Required: `source_id`, `domain` (one of the six train domains), `query`, `document`, `license`,
`synthetic_query` (bool). Optional: `title`, `answer`, `source_url`, `generation_method`;
`eval_only` / `public_benchmark` default to `false`.

## Data types

1. **`faq_real`** — WebFAQ / WebFAQ 2.0 question→answer pairs (CC-BY-4.0).
2. **`qa_passage_non_eval`** — GermanQuAD-*style* question/passage pairs, **never** drawn from
   GermanQuAD test/eval text (declared `eval_only=false`, `public_benchmark=false`, and the text
   passes the leakage index).
3. **`web_nonfaq`** — general web paragraph/document chunks with teacher-generated questions
   (`synthetic_query=true`, `generation_method` set; license inherits the source doc).
4. **`long_doc_chunks`** — documents split into passages with multi-sentence queries.
5. **`local_rag`** — user-provided internal/doc corpus, if present (trained on a split disjoint
   from its eval split).
6. **`german_stress`** — compounds, negation, dates, abbreviations, entity disambiguation.

## CLI

```
python scripts/build_v5_rag_data.py \
  --config configs/experiments/v5_small_rag.json \
  --inputs data/raw/v5/*.jsonl \
  --output data/processed/v5/rag_pairs.jsonl \
  --report outputs/v5-small-rag/data_mixer_report.json \
  --target-count 100000 \
  --max-faq-share 0.35 \
  --min-nonfaq-share 0.50 \
  --dry-run
```

`--dry-run` validates, samples, and writes the **report** (proving non-FAQ coverage before
teacher scoring) but does **not** write the pairs file and imports no torch. Exit code is `1` on
any hard failure in dry-run or real mode, `0` only when the mixture passes every gate.

## Hard-fail gates

1. **Unknown license** — any row whose license is missing/empty/`unknown` or not in the v5
   allowlist (`data.ALLOWED_LICENSES` + `synthetic-inherits-source`).
2. **Public-benchmark / eval leakage** — any row with `public_benchmark=true`, `eval_only=true`,
   or a public-benchmark token (germanquad, dt_test, gerdalir, webfaq, …) in
   `source_id`/`source_url`/`domain`. Text-level provenance is still the job of
   `scripts/run_full_leakage_scan.py`; this gate enforces the declared flags + tokens.
3. **Too FAQ-heavy** — sampled `faq_real` share exceeds `--max-faq-share`.
4. **Too FAQ-poor** — sampled non-FAQ share below `--min-nonfaq-share`.
5. **Schema / config** — bad row schema, or an input domain not in the config's `train_domains`.

## Deterministic, domain-balanced sampling

Rows are keyed by `blake2b(source_id|query|document)` and sampled **round-robin over
alphabetically-sorted domains** (rows within a domain pre-sorted by their stable key). There is
no RNG, so the same inputs always yield the same mixture, **independent of input file order**.
When non-FAQ domains run dry, the remaining budget is filled from whatever rows remain (usually
FAQ) — and the resulting over-FAQ mixture is then rejected by gate 3 rather than shipped. That is
the point: scarce non-FAQ coverage fails loudly.

## Report (`--report`)

Proves non-FAQ coverage before any teacher scoring:

- `rows_by_domain`, `rows_by_source`, `rows_by_license`
- `faq_share`, `nonfaq_share`
- `synthetic_share`, `real_share`
- `query_style_distribution` (keyword / question / statement / multi_sentence)
- `examples_per_domain` (up to 3 per domain)
- `available_by_domain`, `selected_rows`, `status`, `errors`

## Acceptance

- v5 cannot silently become another FAQ-only run — an over-FAQ mixture is a non-zero exit.
- The mixture report proves non-FAQ coverage before teacher scoring is ever launched.
