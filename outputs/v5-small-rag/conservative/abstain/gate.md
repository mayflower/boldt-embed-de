# v5 small-RAG abstention promotion gate: **fail**

| eval set | policy Δ vs first-stage | always_rerank Δ | medium+hard Δ | abstain | catastrophic |
|---|--:|--:|--:|--:|--:|
| webfaq | +0.097458 | 0.134936 | 0.21511 | 0.241429 | 0.008571 |
| germanquad | +0.024292 | 0.009421 | 0.362319 | 0.260667 | 0.074 |
| dt_test | +0.019332 | 0.021205 | 0.495699 | 0.416 | 0.0 |

## Checks

- ✅ webfaq_overall: +0.0975 (min +0.05)
- ✅ webfaq_medium_hard: 0.21511 (min +0.2)
- ✅ germanquad_overall: +0.0243 (min -0.005)
- ❌ germanquad_catastrophic: 0.0740 (max 0.03)
- ✅ germanquad_beats_always_rerank: policy +0.0243 vs always_rerank +0.0094
- ✅ dt_test_overall: +0.0193 (min -0.005)
- ✅ dt_test_catastrophic: 0.0000 (max 0.02)
- ❌ dt_test_beats_always_rerank: policy +0.0193 vs always_rerank +0.0212

**Verdict: NOT promoted** — failing: germanquad_catastrophic, dt_test_beats_always_rerank. Reranker stays Experimental.
