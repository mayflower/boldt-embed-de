# RAG-reranker evaluation schema

A RAG reranker is judged on whether it **promotes answer-supporting passages inside a fixed
top-k candidate set** — not on dense retrieval over a whole corpus. This defines the schemas,
the leakage-safe split, and the metrics. Module: `boldt_embed.rag_eval_schema` · build:
`scripts/build_rag_eval_sets.py` · validate: `scripts/validate_rag_eval_sets.py`. Pure stdlib.

## Schemas

**Query**
```json
{
  "query_id": "...", "query": "...", "answer": "(optional)",
  "positive_doc_ids": ["..."],
  "domain": "faq_real|web|wiki_non_eval|local_rag|...",
  "source": "...",
  "metadata": {
    "requires_answer_support": true,
    "answer_type": "fact|definition|procedure|faq|multi_hop|other"
  }
}
```

**Corpus doc**
```json
{"doc_id": "...", "text": "...", "title": "(opt)", "source": "...", "domain": "...",
 "license": "...", "url": "(opt)"}
```

**Fixed candidate list** (what the reranker actually re-orders)
```json
{"query_id": "...", "query": "...",
 "candidates": [{"doc_id": "...", "text": "...", "first_stage_score": 0.0,
                 "candidate_source": "bm25|dense|e5|qwen|manual|webfaq_hardneg",
                 "label": 1, "teacher_score": 0.0}],
 "positive_doc_ids": ["..."], "domain": "...", "source": "..."}
```

## Building eval sets

- **WebFAQ-style held-out** (`--mode webfaq --faq-input <real FAQ jsonl> --split test`):
  deterministic train/dev/test split by `assign_split(query)` (stable blake2b hash, default
  10%/10%/80%). Same query → same split on every machine, so **train/dev/test never share a
  (query, answer) pair** → leakage-safe.
- **Local RAG** (`--mode local`): reads `data/eval/rag_local/{corpus,queries,qrels}.jsonl` and
  validates them into an eval set.

## Validators

- every `positive_doc_id` appears in the corpus (`validate_eval_set`);
- a candidate list used for reranker lift contains **at least one positive**
  (`validate_candidate_list(require_positive=True)`);
- **public/eval data must never appear in a training candidate file**
  (`check_no_eval_leakage`): any train row whose `query_id` or (candidate) `doc_id` is in the
  eval split is flagged.

```bash
python scripts/validate_rag_eval_sets.py \
  --queries data/eval/webfaq/queries.jsonl --corpus data/eval/webfaq/corpus.jsonl \
  --qrels data/eval/webfaq/qrels.jsonl \
  [--candidate-lists shortlists.jsonl] [--train-candidates train.jsonl]
```

## Metrics

Over the **fixed candidate set** (ranked by reranker score):

- `ndcg@10`, `mrr@10`, `recall@10` (reused from `boldt_embed.metrics`),
- `positive_in_top_10` — a positive is in the top 10,
- `answer_support_at_10` — same, but averaged **only over queries with
  `requires_answer_support=true`** (the RAG question of "is the answer actually retrievable?"),
- `reranker_delta_ndcg10` — nDCG@10 of (first-stage order) → (reranked order) over the same
  fixed candidates. This is the v4 promotion signal (must be ≥+0.03 on WebFAQ/local RAG,
  neutral-or-better on GermanQuAD/DT-test; see `docs/v4-rag-reranker-plan.md`).
