# Frozen bounded-policy promotion gate: **fail**

_Decision is about the policy `bounded_margin_override_v1`, never raw always-rerank. Diagnostic sets ignored: none._

| eval set | role | policy Δ | raw Δ | catastrophic | mode |
|---|---|--:|--:|--:|---|
| dt_test | guardrail | +0.017485 | +0.021205 | 0.0 | policy_gated |
| germanquad | guardrail | +0.036906 | +0.008698 | 0.003333 | policy_gated |
| near_ceiling | primary | -0.000515 | -0.008779 | 0.001397 | policy_gated |
| webfaq | primary | +0.02452 | +0.139584 | 0.000735 | policy_gated |

## Checks

- ✅ not_tuned_on_guardrail: policy must not be tuned on GermanQuAD/DT-test
- ❌ webfaq_policy_delta: +0.0245 (min +0.05)
- ✅ near_ceiling_catastrophic: 0.0014 (max 0.03)
- ✅ near_ceiling_policy_delta: -0.0005 (min -0.005)
- ✅ germanquad_catastrophic: 0.0033 (max 0.03)
- ✅ germanquad_policy_delta: +0.0369 (min -0.005)
- ✅ dt_test_catastrophic: 0.0000 (max 0.02)
- ✅ dt_test_policy_delta: +0.0175 (min -0.005)
- ✅ germanquad_beats_raw_rerank: policy +0.0369 vs raw +0.0087
- ✅ near_ceiling_beats_raw_rerank: policy -0.0005 vs raw -0.0088

**Verdict: NOT promoted** — failing: ['webfaq_policy_delta']. Reranker stays Experimental.
