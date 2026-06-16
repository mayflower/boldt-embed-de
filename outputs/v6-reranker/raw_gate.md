# v6 RAW reranker promotion gate: **fail**

_Decides whether the reranker MODEL is promotable. RAW lift over fixed candidate lists only — no bounded policy / serving wrapper / abstention. Diagnostic (ignored): none._

| eval set | role | mode | Δ nDCG@10 | catastrophic | present |
|---|---|---|--:|--:|--:|
| dt_test | guardrail | raw | +0.003572 | 0.007 | 1.0 |
| germanquad | guardrail | raw | -0.08642 | 0.207333 | 1.0 |
| webfaq | primary | raw | +0.035829 | 0.050127 | 0.965736 |

## Checks

- ✅ no_policy_gated_card: model card must not recommend a policy workaround
- ✅ no_public_eval_leakage: no public-eval leakage in candidate lists
- ❌ webfaq_delta: +0.0358 (min +0.05)
- ✅ webfaq_positive_present: 0.966 (min 0.8)
- ❌ germanquad_delta: -0.0864 (min -0.005)
- ❌ germanquad_catastrophic: 0.2073 (max 0.03)
- ✅ dt_test_delta: +0.0036 (min -0.005)
- ✅ dt_test_catastrophic: 0.0070 (max 0.02)

**Verdict: NOT promoted** — failing: ['webfaq_delta', 'germanquad_delta', 'germanquad_catastrophic'].
