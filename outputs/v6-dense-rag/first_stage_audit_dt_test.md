# First-stage recall audit — dt_test

**Bottleneck: `near_ceiling_first_stage`.** first stage is already near-ceiling; reranking can only churn it. Reranker lift is not the lever here.

_1000 queries, 1000 positives. A reranker can only reorder what the first stage retrieved — injected/oracle sources (e.g. `manual`) are NOT counted as retriever hits._

## Recall & nDCG

| metric | value |
|---|--:|
| recall@10 | 0.992 |
| recall@20 | 0.994 |
| recall@50 | 0.994 |
| recall@100 | 0.994 |
| recall@200 | 0.994 |
| positive_in_top_10 rate | 0.992 |
| first-stage nDCG@10 | 0.977444 |
| oracle nDCG@10 (retriever-only, realistic) | 0.994 |
| oracle nDCG@10 (with injected positives) | 1.0 |
| **upper-bound reranker lift (realistic)** | **0.016556** |
| upper-bound lift IF candidate set were perfect | 0.022556 |
| illusory lift from injected positives | 0.006 |

## Missing positives (reranker can NEVER recover these)

- missing_positive_count: **6** (rate **0.006**)
- by domain: {'unknown': 6}
- by query_style: {'unknown': 6}
- by source-of-missing: {'injected_only': 6}

## Candidate-source contribution

| bucket | positives |
|---|--:|
| BM25 hits | 994 |
| dense hits | 0 |
| other-retriever hits | 0 |
| BM25-only | 994 |
| dense-only | 0 |
| overlap (BM25 ∧ dense) | 0 |
| union (any retriever) | 994 |
| injected-only (NOT retrieved) | 6 |
| absent from list | 0 |

**Has a dense first stage in these lists: False.** No dense candidate source is present — these lists are BM25-only (plus injected gold), so dense-vs-BM25 recall cannot be compared from this data.

## Examples (deterministic)

- `dt2528` [unknown] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `d2528`: “Was ist Ƙ und wo kommt es vor?”
- `dt596` [unknown] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `d596`: “Was waren Mark Forests berufliche Tätigkeiten?”
- `dt2145` [unknown] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `d2145`: “Wer wird in dieser Liste aufgeführt?”
- `dt4111` [unknown] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `d4111`: “Welcher Religion gehört Auslanders Familie an?”
- `dt4918` [unknown] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `d4918`: “Was ist die Adresse des denkmalgeschützten Gebäudes?”
- `dt1660` [unknown] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `d1660`: “Wie lautet der ukrainische Name des Ortes in dem Text?”
