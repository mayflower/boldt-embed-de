# v5 frozen-policy failure analysis

Policy `bounded_margin_override_v1`. **No training until the failure mode is identified.** A query is a failure if the policy regressed it (policy < first-stage) or under-lifted it (raw-rerank beats policy by > 0.02). Pure stdlib, no ML.

Gate-failing sets: **['webfaq']**.

## Per-set failures

| set | role | gate? | queries | failures | rate | policy-fixable | top category |
|---|---|:--:|--:|--:|--:|--:|---|
| dt_test | guardrail | · | 1000 | 7 | 0.007 | 0.5714 | margin_override_too_strict |
| germanquad | guardrail | · | 1500 | 117 | 0.078 | 0.8547 | margin_override_too_strict |
| near_ceiling | primary | · | 716 | 1 | 0.0014 | 1.0 | margin_override_too_permissive |
| webfaq | primary | yes | 1360 | 344 | 0.2529 | 0.3198 | no_useful_first_stage_score |

## Which constraint caused it (by set)

### dt_test — 7 failures (0.7%), 57% policy-fixable
- by category: {'margin_override_too_strict': 3, 'no_useful_first_stage_score': 3, 'top_k_lock_too_strict': 1}
- by type: {'missed_lift': 6, 'regression': 1}
- by domain: {'unknown': 7}
- by displacer source: {'bm25': 7}
- recommended fix mix: {'tune_policy_threshold': 4, 'add_calibration_features': 3}

  - `dt3154` [missed_lift] **margin_override_too_strict** — reranker gap over first-stage top1 = 2.11 < margin 3.0; override did not fire (fs 0.6309 → raw 1.0 → policy 0.6309)
  - `dt2145` [missed_lift] **no_useful_first_stage_score** — no usable first_stage_score → locks built on a meaningless order (fs 0.0 → raw 0.6309 → policy 0.0)
  - `dt4396` [missed_lift] **margin_override_too_strict** — reranker gap over first-stage top1 = 1.96 < margin 3.0; override did not fire (fs 0.6309 → raw 1.0 → policy 0.6309)
  - `dt214` [regression] **top_k_lock_too_strict** — positive at first-stage rank 5 (tail); raw lifts to 4; head locked at top-3 so policy left it at 6 (fs 0.3562 → raw 0.3869 → policy 0.3333)

### germanquad — 117 failures (7.8%), 85% policy-fixable
- by category: {'margin_override_too_strict': 80, 'no_useful_first_stage_score': 17, 'top_k_lock_too_strict': 9, 'margin_override_too_permissive': 5, 'positive_locked_too_low': 5, 'blend_alpha_too_low': 1}
- by type: {'missed_lift': 111, 'catastrophic': 5, 'regression': 1}
- by domain: {'unknown': 117}
- by displacer source: {'bm25': 117}
- recommended fix mix: {'tune_policy_threshold': 100, 'add_calibration_features': 17}

  - `gq658` [missed_lift] **no_useful_first_stage_score** — no usable first_stage_score → locks built on a meaningless order (fs 0.0 → raw 0.5 → policy 0.0)
  - `gq1317` [missed_lift] **margin_override_too_strict** — reranker gap over first-stage top1 = 0.91 < margin 3.0; override did not fire (fs 0.5 → raw 1.0 → policy 0.5)
  - `gq225` [missed_lift] **no_useful_first_stage_score** — no usable first_stage_score → locks built on a meaningless order (fs 0.0 → raw 0.3562 → policy 0.0)
  - `gq518` [missed_lift] **margin_override_too_strict** — reranker gap over first-stage top1 = 1.39 < margin 3.0; override did not fire (fs 0.3869 → raw 1.0 → policy 0.4307)

### near_ceiling — 1 failures (0.1%), 100% policy-fixable
- by category: {'margin_override_too_permissive': 1}
- by type: {'catastrophic': 1}
- by domain: {'faq_real': 1}
- by displacer source: {'bm25': 1}
- recommended fix mix: {'tune_policy_threshold': 1}

  - `q99df527ad767eb91` [catastrophic] **margin_override_too_permissive** — margin override promoted a non-positive to rank 1 (fs 1.0 → raw 0.6309 → policy 0.6309)

### webfaq — 344 failures (25.3%), 32% policy-fixable
- by category: {'no_useful_first_stage_score': 234, 'margin_override_too_strict': 96, 'top_k_lock_too_strict': 10, 'positive_locked_too_low': 2, 'margin_override_too_permissive': 2}
- by type: {'missed_lift': 342, 'catastrophic': 1, 'regression': 1}
- by domain: {'faq_real': 344}
- by displacer source: {'bm25': 344}
- recommended fix mix: {'add_calibration_features': 234, 'tune_policy_threshold': 110}

  - `qa82ceaccb240a11c` [missed_lift] **no_useful_first_stage_score** — no usable first_stage_score → locks built on a meaningless order (fs 0.0 → raw 0.2891 → policy 0.0)
  - `q13a65a84d840fede` [missed_lift] **no_useful_first_stage_score** — no usable first_stage_score → locks built on a meaningless order (fs 0.0 → raw 0.2891 → policy 0.0)
  - `qc7af909b072243fe` [missed_lift] **margin_override_too_strict** — reranker gap over first-stage top1 = 0.55 < margin 3.0; override did not fire (fs 0.6309 → raw 1.0 → policy 0.6309)
  - `q3a4067050d04fced` [missed_lift] **margin_override_too_strict** — reranker gap over first-stage top1 = 0.45 < margin 3.0; override did not fire (fs 0.6309 → raw 1.0 → policy 0.6309)

## First-stage score calibration across sets

Per-set first-stage score spread: `{'dt_test': 73.2864, 'germanquad': 61.7928, 'near_ceiling': 91.2023, 'webfaq': 91.7495}`. Cross-set divergence flagged: **False** (the absolute `margin_override` threshold of 3.0 on reranker logits means different things when score scales differ by set).

## Would adjusting the policy fix it?

Aggregate recommended-fix counts across failures: `{'add_calibration_features': 254, 'tune_policy_threshold': 215}`.

- **215/469** failures map to a tunable bound; **254/469** map to missing first-stage signal (calibration). But the bounds exist to protect the GermanQuAD/near-ceiling guardrails — loosening `preserve_first_stage_top_k`/`max_upshift`/`margin_override` to capture WebFAQ lift is exactly what reintroduces guardrail catastrophic drops. The SAME tunable categories (margin_override/top_k_lock) dominate the GermanQuAD guardrail failures ({'margin_override_too_strict': 80, 'no_useful_first_stage_score': 17, 'top_k_lock_too_strict': 9, 'margin_override_too_permissive': 5, 'positive_locked_too_low': 5, 'blend_alpha_too_low': 1}). A single global bound cannot satisfy both unless the policy can tell the two regimes apart.

## Recommendation

1. **Add calibration features (highest priority).** 234/344 WebFAQ failures are queries where the first stage never retrieved the positive (no first_stage_rank/score) — a policy that bounds *around first-stage order* cannot lift them. Add a per-query first-stage-confidence signal (has-first-stage-signal, score dispersion, positive-retrieved) so the policy can detect this regime and trust the reranker there, while staying bounded where the first stage is confident.
2. **Audit WebFAQ first-stage recall, not the reranker.** Much of the raw lift is just recovering positives BM25 failed to retrieve; that is a retrieval/measurement gap, so part of the +0.05 bar may be unreachable by any reranking policy on these candidate lists.
3. **Do NOT globally tune the bounds.** Calibration divergence across sets is False (not a score-scale problem); the tunable failures help WebFAQ and hurt the guardrails symmetrically. Any threshold change must be calibration-gated (conditional), not global.
4. **Do NOT train a new checkpoint yet** (unknown/reranker-wrong failures are negligible; the reranker scores are usable — the policy just cannot act on them) and **do NOT add more data blindly** (no candidate-source or duplicate artifacts surfaced).

**Conclusion:** the failure mode is identified — WebFAQ under-lift is dominated by missing first-stage signal (structural, not a threshold), and the remaining tunable failures are in direct tension with the guardrails. **No training; next step is calibration features + a WebFAQ first-stage recall audit.**
