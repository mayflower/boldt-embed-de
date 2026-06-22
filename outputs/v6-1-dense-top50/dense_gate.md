# Dense gate (dense-v6.1): **fail**

_Decides whether the Boldt dense RAG embedder can be recommended for German RAG first-stage retrieval — dense retrieval quality only, INDEPENDENT of the reranker._

| check | status | detail |
|---|---|---|
| webfaq_recall_at_50 | ✅ | 0.9334 (min 0.9) |
| webfaq_recall_at_100 | ✅ | 0.9765 (min 0.96) |
| webfaq_missing_positive_rate | ✅ | 0.0114 (max 0.04) |
| webfaq_ndcg_at_10 | ✅ | 0.7044 (min 0.67) |
| germanquad_ndcg_at_10 | ❌ | 0.878 (min 0.88) |
| dt_test_ndcg_at_10 | ✅ | 0.9748 (min 0.94) |
| matryoshka_256_retention | ✅ | 1.0007 (min 0.95) |
| no_public_eval_leakage | ✅ | no public-eval leakage |

**Do NOT recommend the dense embedder yet. Failed dense targets: germanquad_ndcg_at_10 (0.878 (min 0.88))**
