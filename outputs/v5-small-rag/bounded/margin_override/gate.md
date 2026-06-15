# v5 small-RAG abstention promotion gate: **pass**

| eval set | policy Δ vs first-stage | always_rerank Δ | medium+hard Δ | abstain | catastrophic |
|---|--:|--:|--:|--:|--:|
| webfaq | +0.100569 | 0.13604 | 0.215286 | 0.008571 | 0.001429 |
| germanquad | +0.043938 | 0.008698 | 0.29212 | 0.000667 | 0.014667 |
| dt_test | +0.019598 | 0.021205 | 0.502501 | 0.0 | 0.0 |

## Checks

- ✅ webfaq_overall: +0.1006 (min +0.05)
- ✅ webfaq_medium_hard_positive: 0.215286 (must be > 0)
- ✅ germanquad_overall: +0.0439 (min -0.005)
- ✅ germanquad_catastrophic: 0.0147 (max 0.03)
- ✅ germanquad_beats_raw_always_rerank: policy +0.0439 vs always_rerank 0.008698
- ✅ dt_test_overall: +0.0196 (min -0.005)
- ✅ dt_test_catastrophic: 0.0000 (max 0.02)
