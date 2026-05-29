# German Evaluation Suite

Encoder: `HashingEncoder(dim=256) STAND-IN (not Boldt)`

> Default HashingEncoder validates plumbing only; not a Boldt quality claim.

| Task | Metric | Value |
|---|---|---:|
| STS | spearman | 0.8675 |
| Classification | accuracy | 1.0000 |
| Clustering | v_measure | 0.1690 |
| Cross-lingual DE→EN | ndcg@10 | 0.5982 |
| RAG | ndcg@10 | 0.9000 |
| Stress (BM25) | ndcg@10 | 1.0000 |

## Stress by case (BM25)

| Case | recall@1 | ndcg@10 |
|---|---:|---:|
| compound | 1.00 | 1.0000 |
| legal_ref | 1.00 | 1.0000 |
| negation | 1.00 | 1.0000 |
| number_date | 1.00 | 1.0000 |
| orthography | 1.00 | 1.0000 |
| regional | 1.00 | 1.0000 |