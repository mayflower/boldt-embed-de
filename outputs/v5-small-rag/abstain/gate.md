# v5 small-RAG abstention promotion gate: **fail**

| eval set | policy Δ vs first-stage | always_rerank Δ | medium+hard Δ | abstain | catastrophic |
|---|--:|--:|--:|--:|--:|
| webfaq | +0.128456 | 0.173402 | 0.281239 | 0.217143 | 0.008571 |
| germanquad | -0.001553 | -0.028452 | 0.281152 | 0.286667 | 0.103333 |
| dt_test | +0.01799 | 0.021118 | 0.461271 | 0.418 | 0.0 |

## Checks

- ✅ webfaq_overall: +0.1285 (min +0.05)
- ✅ webfaq_medium_hard: 0.281239 (min +0.2)
- ✅ germanquad_overall: -0.0016 (min -0.005)
- ❌ germanquad_catastrophic: 0.1033 (max 0.03)
- ✅ germanquad_beats_always_rerank: policy -0.0016 vs always_rerank -0.0285
- ✅ dt_test_overall: +0.0180 (min -0.005)
- ✅ dt_test_catastrophic: 0.0000 (max 0.02)
- ❌ dt_test_beats_always_rerank: policy +0.0180 vs always_rerank +0.0211

**Verdict: NOT promoted** — failing: germanquad_catastrophic, dt_test_beats_always_rerank. Reranker stays Experimental.
