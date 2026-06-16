# Bounded rerank policy v1 (`bounded_margin_override_v1`)

A **versioned deployment-policy artifact** — `configs/policies/bounded_margin_override_v1.json` —
that pins the v5 conservative checkpoint + the inference-time bounds under which the reranker is
safe to serve. It is **not a new model**. Loader/validator: `src/boldt_embed/policy_config.py`;
validator CLI: `scripts/validate_rerank_policy.py`.

## Why this policy exists

Across v5 we established (real runs, all on disk):
- **raw always-rerank is unsafe** on near-ceiling lists — it churns GermanQuAD (catastrophic 0.11–0.18
  across every conservative checkpoint, incl. the stronger-preservation grid lp04/lp06/lp08);
- **bounded `margin_override` is safe** — on the original conservative checkpoint it cuts GermanQuAD
  catastrophic to **0.015** and passes the full 7-check gate (WebFAQ +0.10, GermanQuAD +0.044,
  DT-test +0.020);
- the catastrophic-drop analysis found **100% of the failures are policy-fixable**, and more
  training did **not** change the deployable answer.

So the deployable unit is *checkpoint + bounded policy*, and we must prevent anyone from accidentally
shipping the raw always-reranker. This artifact encodes that.

## Why raw reranking is unsafe (here)

GermanQuAD/DT-test first stages are near-ceiling (oracle ~1.0). A reranker can only churn an
already-correct top order on those lists; on short factoid queries it demotes the correct long
passage out of the top-10 (the dominant `query_style_mismatch` error). Raw always-rerank therefore
produces a high catastrophic-drop rate even when its average lift looks fine.

## How bounded margin_override works

Reorder the fixed candidate list **only within bounds** that protect a confident first stage, using
**observable features only** (`first_stage_rank/score`, `reranker_score`, `candidate_source`,
counts) — never qrels/labels/oracle/hardness/eval-set name:

- keep the first-stage **top-`preserve_first_stage_top_k`** (3) unless overridden;
- a doc may not drop more than `max_downshift` (2) or rise more than `max_upshift_without_margin`
  (5) ranks unless its reranker score beats the locked doc by `margin_override` (3.0);
- blend `alpha·first_stage_norm + (1−alpha)·reranker_norm` with `blend_alpha_high_confidence` (0.85)
  on confident lists, `blend_alpha_default` (0.65) otherwise.

The override margin is what lets the reranker make *justified* corrections (real lift on medium/hard
lists) while refusing unjustified churn on near-ceiling ones.

## How to serve it

```bash
python scripts/validate_rerank_policy.py \
  --policy configs/policies/bounded_margin_override_v1.json \
  --require-model-exists --format markdown
```

Load via `policy_config.load_policy(...)`, score the fixed candidate list with the pinned
`model_checkpoint`, then apply the bounded policy (`bounded_rerank.apply_policy`) using the
artifact's `bounds`. Re-run the v5 gate on `validation.required_eval_sets` before promotion; the
artifact ships **not promoted** until the policy is frozen and validated on a held-out
`near_ceiling_heldout` guardrail.

## What it must NOT claim

- **Not** a recommendation for **raw always-rerank** — `raw_always_rerank_recommended` is `false` and
  validation FAILS if it is ever flipped to true (or `recommended_mode` set to a raw mode).
- **Not** a dense retriever / standalone retrieval — re-ranks a fixed candidate list only.
- **Not** legal advice. (`applicability.not_for`.)
- Inference must use only `features_allowed_at_inference`; validation FAILS if any
  `features_forbidden_at_inference` (qrels/labels/oracle/hardness/eval-set/positive_doc_ids) leaks
  into the allowed set.

## Validation guards (enforced by `validate_policy`)

Fails if: `raw_always_rerank_recommended` is true; forbidden ∩ allowed inference features non-empty;
`model_checkpoint` missing; no `bounds`; or any `validation` threshold
(`max_germanquad_catastrophic`, `min_webfaq_delta_ndcg10`, `min_dt_test_delta_ndcg10`) missing.
