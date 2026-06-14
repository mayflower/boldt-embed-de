# Scalable lexical mining (BM25 index)

The v2 mining bottleneck: `eval_harness.bm25_rank` re-tokenizes the **entire corpus on every
query**, so `mine_bm25_candidates` was effectively O(n_queries × n_corpus). At v2 scale that
forced hard-negative and reranker-candidate mining onto a logged ~3.5k subset. v3 must mine over
the **full** candidate/corpus set.

## The fix: build the inverted index once

`src/boldt_embed/bm25_index.py` builds an Okapi-BM25 **inverted index** up front, then answers
each query by touching only the postings of the query's terms — O(Σ query-term postings), not
O(corpus). Building once and serving N queries is the difference between hours and seconds.

```python
from boldt_embed.bm25_index import build_bm25_index, BM25Index
idx = build_bm25_index(corpus)                 # corpus: [{doc_id, text}, ...]  — built ONCE
hits = idx.search("Wie hoch darf die Mietkaution sein?", top_k=50)   # [(doc_id, score), ...]
results = idx.batch_search(queries, top_k=50)  # same result as per-query search(); no rebuild
idx.save("index.json"); idx2 = BM25Index.load("index.json")          # reuse across runs
```

- **Tokenization** (`tokenize_de`): lowercase + `ß`→`ss` (always); umlaut folding
  (`ä`→`ae`, `ö`→`oe`, `ü`→`ue`) is **optional** (`fold_umlauts`, recorded in the index so
  query-time matches index-time). Document ids are preserved and returned by search.
- **Deterministic**: ties broken by `doc_id` ascending.
- Pure stdlib, no ML, no network.

## CLIs

```bash
# Build once over a corpus (id/text field names configurable):
python scripts/build_bm25_index.py --corpus corpus.jsonl --output index.json \
  --text-field text --id-field doc_id [--fold-umlauts]

# Batch-search a prebuilt index (loaded once, reused for every query):
python scripts/search_bm25_index.py --index index.json --queries queries.jsonl \
  --top-k 10 --output results.jsonl
```

## Mining over the full set

`mine_hard_negatives_2026.py` and `build_reranker_candidates_v2.py` both:

- accept `--bm25-index <index.json>`; if absent they build the index **once** (never per query);
- log the full `corpus=… queries=…/… cap_applied=…` line;
- emit v3 report fields into the printed stats and a `<output>.mining_report.json` sidecar:
  **`mining_corpus_size`**, **`mining_query_count`**, **`mining_cap_applied`**,
  **`bm25_runtime_sec`**;
- support `--max-queries N` for an **explicit** subsample (sets `mining_cap_applied=true`), and
  `--require-full-corpus`, which **fails (exit 2)** if any cap/subsample is applied.

`negative_mining_2026.mine_bm25_candidates(queries, corpus, k, index=None)` builds the index once
(or reuses a passed-in `BM25Index`) — this is the single code path both scripts use.

## Anti-cap policy (promotion gate)

Any capped/subsampled mining must be **explicit and visible**: it only happens via `--max-queries`
and is recorded as `mining_cap_applied: true` in the report. A v3/release run uses
`--require-full-corpus` so a cap is a hard failure, and the `mining_cap_applied` field lets the
release/promotion gate refuse a model whose negatives were mined on a subset. See
`docs/hard-negative-mining-2026.md` and `docs/v3-real-domain-generalization-plan.md`.
