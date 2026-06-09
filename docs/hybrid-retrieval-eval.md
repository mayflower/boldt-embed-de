# Hybrid retrieval evaluation

Production German retrieval needs **sparse and dense**, not one or the other. BM25 nails exact
terms German throws at you — legal references (`§ 551`), long compounds, rare named entities —
which dense models tend to smear; dense recovers paraphrase and morphological variation BM25
misses. This harness evaluates BM25, dense, their RRF fusion, and fusion + reranker, plus a
Matryoshka dimension sweep. Code: `src/boldt_embed/hybrid_eval.py`,
`scripts/eval_hybrid_retrieval.py`.

## Modes

| mode | pipeline |
|---|---|
| `bm25_only` | lexical BM25 |
| `dense_only` | embedder cosine |
| `hybrid_rrf` | reciprocal-rank fusion of BM25 + dense |
| `hybrid_rrf_plus_reranker` | fuse, then rerank the top-`k` with the cross-encoder |

Reciprocal-rank fusion: `score(d) = Σ_r 1/(k + rank_r(d))` (default `k=60`), deterministic
with first-appearance tie-breaks. The reranker only re-orders the fused head (`--top-k-rerank`).

## Metrics

Per query, aggregated across queries: **nDCG@10, MRR@10, Recall@10, Recall@100, MAP@10,
positive-in-top-10**. With a real model run, encode **throughput** (texts/sec) is reported too.

## Matryoshka dimension sweep

For each dim in `--dims` (default `1024,768,512,256,128,64`): truncate every embedding to the
prefix, **re-normalize**, dense-rank, and report the full metric set. This shows the
accuracy/footprint trade-off so production can pick the smallest dim that holds quality.

## Run it

```bash
# Offline: validate inputs + BM25-only numbers + planned modes/dims (no model, no torch)
python scripts/eval_hybrid_retrieval.py \
  --eval-corpus tests/fixtures/hybrid_corpus.jsonl \
  --eval-queries tests/fixtures/hybrid_queries.jsonl \
  --qrels tests/fixtures/hybrid_qrels.jsonl --dims 1024,256,64 --dry-run

# Full run (GPU): all modes + Matryoshka sweep
python scripts/eval_hybrid_retrieval.py \
  --embedder-model outputs/checkpoints/boldt-modern-bi \
  --reranker-model outputs/checkpoints/boldt-reranker-modern \
  --eval-corpus data/eval/gerdalir_corpus.jsonl \
  --eval-queries data/eval/gerdalir_queries.jsonl \
  --qrels data/eval/gerdalir_qrels.jsonl \
  --dims 1024,768,512,256,128,64 --top-k-first-stage 200 --top-k-rerank 50 \
  --output outputs/eval/hybrid_eval.json
```

### Local fixtures for GerDaLIR / GermanQuAD

Convert a benchmark to the three JSONL inputs (`{doc_id,text}`, `{query_id,query}`,
`{query_id,doc_id,relevance}`) once, then point the script at them. Keep these eval fixtures
**out of training** (see the eval-only rule in `docs/data/training-datasets-research-2026.md`).

## First-stage vs reranked lift

Because `hybrid_rrf` and `hybrid_rrf_plus_reranker` run on the *same* fused first stage, the
nDCG@10 delta between them is the reranker's contribution — the honest way to read reranker
value (and the lesson from the v1 reranker that looked fine until measured as lift). All paths
write full run metadata; the dry-run path computes real BM25 numbers offline.
