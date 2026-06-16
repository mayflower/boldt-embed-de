# v5 frozen-policy failure analysis

The frozen `bounded_margin_override_v1` policy **failed its promotion gate on exactly one check**:
`webfaq_policy_delta = +0.0245 < +0.05`. Every guardrail passed — including the held-out near-ceiling
set (policy Δ −0.0005, catastrophic 0.0014). Per the workflow, **no training happens until the
failure mode is identified.** This document identifies it.

- Analyzer: `scripts/analyze_policy_gate_failures.py` (pure stdlib, no ML)
- Machine report: `outputs/v5-small-rag/policy/failure_analysis.{md,json}`
- A query is a **failure** if the frozen policy *regressed* it (policy nDCG@10 < first-stage) or
  *under-lifted* it (raw-rerank beats policy by > 0.02 and raw > first-stage).

> Note: the gate's "near-ceiling validation" did **not** fail (1/716 catastrophic). The failing
> condition is the WebFAQ lift bar. So the analysis targets the WebFAQ under-lift, with the
> guardrails analysed alongside to expose the safety/lift tension.

## Failures by set / domain / source

| set | role | gate-failing | queries | failures | rate | policy-fixable share | top category |
|---|---|:--:|--:|--:|--:|--:|---|
| **webfaq** | primary | **yes** | 1360 | **344** | 0.253 | **0.32** | `no_useful_first_stage_score` |
| germanquad | guardrail | no | 1500 | 117 | 0.078 | 0.85 | `margin_override_too_strict` |
| dt_test | guardrail | no | 1000 | 7 | 0.007 | 0.57 | `margin_override_too_strict` |
| near_ceiling | primary | no | 716 | 1 | 0.001 | 1.00 | `margin_override_too_permissive` |

- **By domain:** WebFAQ/near-ceiling failures are all `faq_real`; GermanQuAD/DT-test failures carry no
  domain label (`unknown`).
- **By displacer source:** the doc that displaced the positive is **`bm25` in 100% of failures
  across every set** — there is **no candidate-source artifact** (no over-represented `manual`/dense
  source) and **no duplicate/near-duplicate confusion** (0 detected). That rules out taxonomy
  categories 7 and 8 as drivers.

## Which policy constraint caused each failure

**WebFAQ (the failing set), 344 failures:**

| category | count | constraint | policy-fixable? |
|---|--:|---|:--:|
| `no_useful_first_stage_score` | **234** | first stage never retrieved the positive | **no** |
| `margin_override_too_strict` | 96 | `bounds.margin_override = 3.0` too high | yes |
| `top_k_lock_too_strict` | 10 | `preserve_first_stage_top_k` / `max_upshift` | yes |
| `positive_locked_too_low` | 2 | intra-head freeze | yes |
| `margin_override_too_permissive` | 2 | margin too low (regression) | yes |

**68% of WebFAQ failures are `no_useful_first_stage_score`** — verified to mean *the positive itself*
was never retrieved by the first stage (no `first_stage_rank`/`first_stage_score`; first-stage
nDCG@10 = 0.0). A policy that bounds *around the first-stage order* structurally **cannot** lift a
positive the first stage never surfaced. Raw reranking can (it has a `reranker_score`), which is
exactly the lift the policy "leaves on the table".

**GermanQuAD guardrail, 117 failures:** dominated by `margin_override_too_strict` (80) and
`top_k_lock_too_strict` (9) — the **same tunable categories** that dominate WebFAQ's *fixable*
failures. Only 5 are `margin_override_too_permissive` and 5 catastrophic.

**near_ceiling guardrail, 1 failure:** a single `margin_override_too_permissive` catastrophic — the
override fired on an already-perfect list and promoted a non-positive (`q99df527…`, fs 1.0 → policy
0.63).

### Examples
- `qa82ceaccb240a11c` [WebFAQ, missed_lift] **no_useful_first_stage_score** — positive not retrieved;
  fs 0.0 → raw 0.289 → **policy 0.0**.
- `qc7af909b072243fe` [WebFAQ, missed_lift] **margin_override_too_strict** — reranker gap over
  first-stage top-1 = 0.55 < margin 3.0; override didn't fire; fs 0.631 → raw 1.0 → **policy 0.631**.
- `gq1317` [GermanQuAD, missed_lift] **margin_override_too_strict** — gap 0.91 < 3.0; fs 0.5 → raw 1.0
  → policy 0.5.
- `q99df527ad767eb91` [near_ceiling, **catastrophic**] **margin_override_too_permissive** — override
  promoted a non-positive on a ceiling list; fs 1.0 → **policy 0.63**.

## Would adjusting the policy fix it?

**Partly, and not safely as a global change.** Of 469 total failures, 215 map to a tunable bound and
254 to missing first-stage signal. Two findings make global tuning the wrong move:

1. **The dominant WebFAQ failure is not a threshold.** 234/344 are positives the first stage never
   retrieved — no bound change rescues them, because the policy trusts an order that doesn't contain
   the positive.
2. **The tunable failures are in direct, symmetric tension with the guardrails.**
   `margin_override_too_strict` dominates the *fixable* WebFAQ failures (96) **and** the GermanQuAD
   guardrail failures (80). Lowering `margin_override` lifts both — but the near-ceiling catastrophic
   is `margin_override_too_permissive`, the opposite direction: a lower margin makes ceiling lists
   *worse*. Cross-set first-stage score-scale divergence is **False** (spreads 62–92 are comparable),
   so this is **not** a simple recalibration — it is a genuine regime difference (confident vs.
   absent first stage) that a single global bound cannot straddle.

## Recommendation

1. **Add calibration features (highest priority).** Give the policy a per-query first-stage-confidence
   signal — `has_first_stage_signal` (was the positive even retrieved), score dispersion, top-1
   margin — so it can *detect* the "no usable first-stage order" regime (234 WebFAQ failures) and the
   "already-solved" regime (near-ceiling), and make the margin/lock **conditional** on confidence
   instead of global. This addresses the structural failure and resolves the tension in #2 of the
   previous section.
2. **Audit WebFAQ first-stage recall, not the reranker.** 469/1360 WebFAQ positives are injected
   `manual` candidates BM25 never retrieved; much of raw's headline +0.14 lift is just recovering
   first-stage recall misses. Part of the +0.05 bar may be **unreachable by any reranking policy** on
   these candidate lists — fix the retriever / candidate generation, or re-scope the WebFAQ target.
3. **Do NOT globally tune the bounds** (see tension above) — any threshold change must be
   calibration-gated.
4. **Do NOT train a new checkpoint yet** — `unknown`/reranker-wrong failures are negligible; the
   reranker scores are usable, the policy simply cannot act on them. **Do NOT add more data blindly**
   — no candidate-source or duplicate artifacts surfaced.

## Conclusion (failure mode identified — gate to "no more training")

The WebFAQ under-lift is **dominated (68%) by missing first-stage signal** — a structural limit of a
bounded-over-first-stage policy, not a threshold — and the remaining tunable failures are in direct
tension with the guardrails. **Therefore: no training.** The identified next step is **calibration
features (conditional bounding) + a WebFAQ first-stage recall audit**, after which a calibration-gated
policy can be re-evaluated against the same frozen gate.
