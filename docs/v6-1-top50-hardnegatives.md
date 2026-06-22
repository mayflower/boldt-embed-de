# v6.1 — dense top-50 hard negatives (Recall@50)

Mines **dense-specific** hard negatives so the v6.1 dense retriever pulls positives from ranks 51–200
up into the top-50. **DENSE-ONLY — this produces training data for the dense embedder; it does NOT
train the reranker.** (Dense-v6: Recall@100 0.964 but Recall@50 0.883 — the positive is often
retrieved but ranked just below the top-50 cutoff.)

- Core: `src/boldt_embed/dense_top50_mining.py` — CLI: `scripts/mine_v6_1_dense_top50_hardnegatives.py`
- Output: `data/processed/v6_1/dense_top50_hardnegatives.jsonl` (+ report)
- Tests: `tests/test_dense_top50_mining.py`

## What it mines (per query whose positive is at dense rank 51..window)

1. **`dense_top50_false_positive`** — the docs dense-v6 ranks **above** the positive. These are the
   blockers; demoting them is exactly what lifts the positive into the top-50. (Primary signal.)
2. **`teacher`** — high-dense-rank docs in the window **below** the positive that Qwen3-Reranker-8B
   confirms are clearly non-relevant (`positive_teacher − doc_teacher ≥ veto_margin`).
3. **`bm25`** — high-BM25 lexical confusions not relevant (`--with-bm25`).
4. **false-negative veto** — drop any candidate whose teacher score is **within `veto_margin`** of
   the positive's: it may actually be relevant (a qrels miss), so it is never trained as a negative.

Negatives are ordered highest-blocker-first (smallest dense rank) and capped at `--max-negatives`.

## Inputs

- Dense-v6 retrieval over the corpus (computed by the CLI; `--dry-run` uses precomputed
  `dense_ranked` lists and imports no ML).
- WebFAQ train `queries` + `qrels` + `corpus` (`outputs/v4-rag-reranker/train/webfaq/`).
- Existing Qwen3-Reranker-8B teacher scores
  (`data/processed/v6/reranker_train_lists_teacher_scored.jsonl`) — **no new 8B run**; used only
  where (query, doc) was already scored, for the veto + teacher hard negatives + margins.
- Optional BM25/e5 retrieval (`--with-bm25`).

## Output schema

```json
{"query_id":"...","query":"...","positive_doc_id":"...","positive":"...","positive_rank_v6":73,
 "negatives":[{"doc_id":"...","text":"...","negative_rank_v6":12,
   "source":"dense_top50_false_positive|teacher|bm25","teacher_score":null,"margin_to_positive":null}],
 "domain":"faq_real|web_nonfaq|...","source":"webfaq_train"}
```

## Report

`outputs/v6-dense-rag/v6_1_top50_hardneg_report.json`: queries mined; **positive_rank_51_100** /
**positive_rank_101_200** counts; total + avg negatives/query; **false_negative_veto_count**;
negatives by source; **teacher-margin distribution**; public-eval `leakage_excluded` count.

## Leakage

Public-eval queries (GermanQuAD/DT-test/…) are **excluded** (`leakage_reason`), so no public-eval
data leaks into v6.1 dense training. Mining runs over WebFAQ **train** only.

## Run

```bash
python scripts/mine_v6_1_dense_top50_hardnegatives.py \
  --corpus  outputs/v4-rag-reranker/train/webfaq/corpus.jsonl \
  --queries outputs/v4-rag-reranker/train/webfaq/queries.jsonl \
  --qrels   outputs/v4-rag-reranker/train/webfaq/qrels.jsonl \
  --output  data/processed/v6_1/dense_top50_hardnegatives.jsonl --with-bm25
# --dry-run mines from precomputed dense_ranked lists (no ML).
```

Validated by `configs/experiments/v6_1_dense_top50.json` (`reranker_training_enabled: false`).
**No reranker training is triggered.**
