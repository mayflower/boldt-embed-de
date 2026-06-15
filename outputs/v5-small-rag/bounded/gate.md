# v5 small-RAG abstention promotion gate: **fail**

| eval set | policy Δ vs first-stage | always_rerank Δ | medium+hard Δ | abstain | catastrophic |
|---|--:|--:|--:|--:|--:|
| webfaq | +0.135513 | 0.13604 | 0.301701 | 0.008571 | 0.014286 |
| germanquad | +0.018217 | 0.008698 | 0.419993 | 0.000667 | 0.103333 |
| dt_test | +0.021205 | 0.021205 | 0.553175 | 0.0 | 0.001 |

## Checks

- ✅ webfaq_overall: +0.1355 (min +0.05)
- ✅ webfaq_medium_hard_positive: 0.301701 (must be > 0)
- ✅ germanquad_overall: +0.0182 (min -0.005)
- ❌ germanquad_catastrophic: 0.1033 (max 0.03)
- ✅ germanquad_beats_raw_always_rerank: policy +0.0182 vs always_rerank 0.008698
- ✅ dt_test_overall: +0.0212 (min -0.005)
- ✅ dt_test_catastrophic: 0.0010 (max 0.02)

**Verdict: NOT promoted** — failing: germanquad_catastrophic. Reranker stays Experimental.
