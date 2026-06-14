# RAG candidate lists

A RAG reranker must learn to **reorder a real first-stage top-k**, not classify random pairs.
This builds fixed candidate lists by merging several first stages, so v4 train/eval runs on
realistic top-k sets. Module: `boldt_embed.rag_candidates` · CLI:
`scripts/build_rag_candidate_lists.py`. Pure stdlib.

## Inputs

- queries / corpus / qrels (see `docs/rag-eval-schema.md`),
- first-stage results, one file per source: `--bm25-results`, `--dense-results` (v3 causal),
  `--e5-results`, `--qwen-results` (each `{query_id, results|candidates|doc_ids:[doc_id|{doc_id,score}]}`),
- optional `--webfaq-hardnegs` (`{query_id, doc_ids:[...]}`),
- optional `--teacher-scores` (`{query_id, doc_id, reranker_score}` — for train labels).

## Output (candidate-list row)

```json
{
  "query_id": "...", "query": "...", "positive_doc_ids": ["..."],
  "candidates": [
    {"doc_id": "...", "text": "...",
     "candidate_source": "bm25|v3_dense|e5_dense|qwen_dense|webfaq_hardneg",
     "first_stage_rank": 0, "first_stage_score": 0.0, "teacher_score": 0.0,
     "label": 1, "domain": "..."}
  ],
  "domain": "...", "source": "..."
}
```

Written to `data/processed/v4/rag_reranker_{train,eval_webfaq,eval_germanquad,eval_dt_test,eval_local}_lists.jsonl`
(one invocation per set; `--mode train|eval`, `--output <path>`).

## Behaviour

1. **Multi-source top-k** — merges each source's top-`--top-k` per query.
2. **Source preserved** — each candidate keeps its `candidate_source`; merge priority is
   bm25 → v3_dense → e5_dense → qwen_dense → webfaq_hardneg → manual.
3. **Dedup** — by `doc_id` and by normalized **text hash** (same passage under two ids collapses);
   the first (priority) source that surfaced a doc keeps its tag.
4. **Positives** — eval lists with no first-stage positive get the gold positive **injected**
   (`candidate_source="manual"`) so lift is scorable, and the query is reported under
   `injected_positive_queries`; train lists with no positive are **skipped** and reported under
   `missing_positive_queries`.
5. **Train labels** — gold positive → 1; teacher-scored → high-precision `v3_label`
   (≥4 → 1, ≤2 → 0, uncertain → **null**, listwise-only). Hard negatives: WebFAQ hard negs +
   BM25 lexical confusions + dense semantic confusions + German-stress adversarial confusions.
6. **Eval labels** — always **null**; positives come from qrels at scoring time (never fabricate
   train labels on public eval).

## Report (first-stage recall is visible)

`candidates_per_query` (min/max/mean), `positive_in_top_k_rate`,
`candidate_source_distribution`, `domains`, and `missing_positive_queries` /
`injected_positive_queries`. A low `positive_in_top_k_rate` means the first stage — not the
reranker — is the bottleneck.

## CLI

```bash
python scripts/build_rag_candidate_lists.py \
  --queries data/eval/webfaq/queries.jsonl --corpus data/eval/webfaq/corpus.jsonl \
  --qrels data/eval/webfaq/qrels.jsonl \
  --bm25-results bm25.jsonl --dense-results v3_dense.jsonl \
  [--e5-results e5.jsonl --qwen-results qwen.jsonl --webfaq-hardnegs hn.jsonl --teacher-scores ts.jsonl] \
  --mode eval --output data/processed/v4/rag_reranker_eval_webfaq_lists.jsonl
```
