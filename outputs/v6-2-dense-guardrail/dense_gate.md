# Dense gate (dense-v6.2): **pass**

_Decides whether the Boldt dense RAG embedder can be recommended for German RAG first-stage retrieval — dense retrieval quality only, INDEPENDENT of the reranker._

| check | status | detail |
|---|---|---|
| webfaq_recall_at_50 | ✅ | 0.9346 (min 0.9) |
| webfaq_recall_at_100 | ✅ | 0.9791 (min 0.96) |
| webfaq_missing_positive_rate | ✅ | 0.0076 (max 0.04) |
| webfaq_ndcg_at_10 | ✅ | 0.7012 (min 0.67) |
| germanquad_ndcg_at_10 | ✅ | 0.8814 (min 0.88) |
| dt_test_ndcg_at_10 | ✅ | 0.9748 (min 0.94) |
| matryoshka_256_retention | ✅ | 0.9981 (min 0.95) |
| no_public_eval_leakage | ✅ | no public-eval leakage |

**Dense embedder CAN be recommended for German RAG first-stage retrieval. The reranker remains experimental unless the raw reranker gate passes (independent decision).**
