# v2 results — verdict: **MIXED** (3/5 criteria)

Reranker promotion gate: **fail**

## Dense retrieval nDCG@10 (best student per dataset)

| dataset | v1 | v2 | min |
|---|---:|---:|---:|
| germanquad | 0.8831334574978987 | 0.8857504605906706 | 0.88 |
| dt_test | 0.9501286162324378 | 0.9442777883635152 | 0.95 |
| gerdalir | 0.07818485477726753 | 0.10958224482523524 | 0.1 |

## Reranker lift (delta over first stage)
- DT-test: 0.0348
- GermanQuAD: -0.0397 (target 0.02)

## Matryoshka 256-d retention: 0.9719 (min 0.95)

## Recommendations
- dense_dt_test_ndcg10: 0.9442777883635152 < target 0.95 — scale/diversify v2 data or retrain.
- reranker_germanquad_delta: -0.0397 < target 0.0 — scale/diversify v2 data or retrain.
- reranker promotion gate FAILED (degrades a held-out set) — do not promote; train on more diverse candidate lists.
