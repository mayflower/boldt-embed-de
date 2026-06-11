# Teacher-driven hard-negative mining (2026)

Hard negatives make or break contrastive retrieval training — and they sank the v1 reranker.
This pipeline mines negatives from **multiple sources**, then uses **teacher scores** to keep
the hard-but-plausible ones and drop likely **false negatives**, balanced across domains.

`src/boldt_embed/negative_mining_2026.py` + `scripts/mine_hard_negatives_2026.py`. Pure
stdlib; dense mining consumes *precomputed* embeddings, so no model is loaded here.

## Why same-domain-only mining is dangerous

The v1 failure: negatives were mined by a weak warmup embedder over a single domain
(Wikipedia). Two failure modes followed:

1. **Too-easy negatives** — the model only learned "relevant vs. obviously-unrelated", never
   "relevant vs. confusingly-similar", so it couldn't separate the top of a real shortlist.
2. **False negatives** — a "negative" that actually answers the query teaches the model to
   *push away a correct passage*. Single-domain mining maximizes this risk because near-
   duplicates abound.

The fix is multi-source mining + a teacher acting as the relevance oracle that vetoes false
negatives, plus domain balancing so no single source dominates.

## Pipeline

```
mine_bm25_candidates              ─┐
mine_dense_candidates_from_embeddings ─┤→ merge_candidate_pools (union, source-labelled)
                                       │      → filter_false_negatives (teacher-gated)
                                       │          → select_domain_balanced_negatives
                                       │              → build_triplets_or_lists
```

- **Multi-source pools.** BM25 (lexical) always; dense (semantic) when embeddings are
  supplied. Merged and deduped, each candidate keeps the source that first surfaced it.
- **False-negative filter.** For each candidate negative, compare its teacher score to the
  *positive's* teacher score (reranker score preferred, else embedding cosine):
  - `neg_score >= pos_score` → drop, reason `neg_score_ge_positive`
  - `neg_score >= pos_score - margin` → drop, reason `within_margin_of_positive`
  - otherwise keep (`false_negative_filter_reason = null`)
  If a score is missing we **keep** the candidate — we never drop blindly.
- **Domain balance.** Cap negatives per domain so one source can't dominate a query's list.
- **Hardest first.** Survivors are ordered by teacher score (highest = closest to the
  positive = hardest), doc_id as a deterministic tie-break, then truncated to
  `--negatives-per-query`.

## Output schema

```json
{"query_id": "q1", "query": "...", "positive_doc_id": "d1", "positive": "...",
 "negatives": [{"doc_id": "d2", "document": "...", "source": "bm25", "domain": "admin",
                "embedding_teacher_score": 0.41, "reranker_teacher_score": 0.40,
                "false_negative_filter_reason": null}],
 "source": "...", "domain": "..."}
```

## Logged counts

Every run prints `total_candidates`, `kept`, `dropped_by_reason`, `kept_by_source`,
`kept_by_domain` — so you can see how many false negatives were vetoed and whether the kept
set is domain-balanced. (No silent truncation.)

## Example: warmup → teacher score → mine → train

```bash
# 0. (warmup embedder + build candidate positives — Prompt 3)
# 1. teacher-score the candidate (query, doc) pairs (GPU)
python scripts/build_teacher_cache.py --input data/processed/candidates.jsonl \
  --output outputs/teacher-cache/teacher_scores.jsonl --mode both

# 2. mine hard negatives, gated by the teacher cache
python scripts/mine_hard_negatives_2026.py \
  --candidates data/processed/candidates.jsonl \
  --corpus data/processed/corpus.jsonl \
  --teacher-cache outputs/teacher-cache/teacher_scores.jsonl \
  --output data/processed/hard_negatives.jsonl \
  --negatives-per-query 8 --false-negative-margin 0.1 --max-per-domain 4

# 3. train the student / reranker on positives + mined hard negatives (Prompts 4, 7)
```

Without `--teacher-cache`, false-negative filtering is disabled and all candidates are kept
(useful for a quick BM25-only smoke run). Margin scale depends on which teacher score is
present: cosine (embedding) lives in `[-1, 1]`; raw reranker logits are unbounded — tune
`--false-negative-margin` to the score you cached.

## v2: multi-source candidate lists

For v2, in addition to the embedder hard-negative file (`hardneg_v2.jsonl`, the schema above),
`scripts/build_reranker_candidates_v2.py` produces **reranker candidate lists**
(`reranker_train_v2.jsonl`) — per query, the positive plus teacher-filtered negatives from
**multiple sources** (BM25 + dense student/e5/teacher), each with `teacher_score`,
`candidate_source`, `domain`. Mixing sources is the fix for the v1 reranker's
single-distribution overfit (see `docs/reranker-training-2026.md`).
