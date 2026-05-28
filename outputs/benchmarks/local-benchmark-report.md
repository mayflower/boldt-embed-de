# Local German Retrieval Benchmark Report

Status: **pass**
Benchmark: `toy_de_retrieval.json`

> This local benchmark validates metric/Matryoshka plumbing only with a BM25 baseline and a deterministic hashing stand-in. It is NOT evidence of Boldt embedding quality.

## BM25 baseline — aggregate

| Metric | Value |
|---|---:|
| map@1 | 1.0000 |
| map@10 | 1.0000 |
| map@3 | 1.0000 |
| map@5 | 1.0000 |
| mrr@1 | 1.0000 |
| mrr@10 | 1.0000 |
| mrr@3 | 1.0000 |
| mrr@5 | 1.0000 |
| ndcg@1 | 1.0000 |
| ndcg@10 | 1.0000 |
| ndcg@3 | 1.0000 |
| ndcg@5 | 1.0000 |
| recall@1 | 1.0000 |
| recall@10 | 1.0000 |
| recall@3 | 1.0000 |
| recall@5 | 1.0000 |

## Hashing stand-in — aggregate

| Metric | Value |
|---|---:|
| map@1 | 1.0000 |
| map@10 | 1.0000 |
| map@3 | 1.0000 |
| map@5 | 1.0000 |
| mrr@1 | 1.0000 |
| mrr@10 | 1.0000 |
| mrr@3 | 1.0000 |
| mrr@5 | 1.0000 |
| ndcg@1 | 1.0000 |
| ndcg@10 | 1.0000 |
| ndcg@3 | 1.0000 |
| ndcg@5 | 1.0000 |
| recall@1 | 1.0000 |
| recall@10 | 1.0000 |
| recall@3 | 1.0000 |
| recall@5 | 1.0000 |

## Hashing stand-in — Matryoshka (nDCG@10 by dim)

| Dim | nDCG@10 | Recall@5 |
|---:|---:|---:|
| 256 | 1.0000 | 1.0000 |
| 128 | 1.0000 | 1.0000 |
| 64 | 1.0000 | 1.0000 |

## Stress-case coverage

- compound: 1
- legal_ref: 1
- negation: 1
- number_date: 1
- orthography: 1
- regional: 1