# v4 RAG reranker promotion gate

Status: **FAIL**

| eval set | delta_ndcg@10 | diagnostic |
|---|--:|:--:|
| dt_test | -0.0007 |  |
| germanquad | -0.0711 |  |
| webfaq | +0.2907 |  |

**Failing:**
- ❌ germanquad_neutral_or_better: -0.0711 (min 0.0)
- ❌ dt_test_neutral_or_better: -0.0007 (min 0.0)
- ❌ germanquad_not_catastrophic: -0.0711 (min -0.02)
