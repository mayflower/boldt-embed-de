# Rerank-or-abstain policy (v5)

A **production-feasible** wrapper around the EXISTING v5 reranker scores: abstain (keep the first
stage) on confident / near-ceiling lists, rerank only uncertain ones. No new model, no training.
Pure stdlib.

- Module: `src/boldt_embed/rerank_abstain.py`
- Fit: `scripts/fit_rerank_abstain_policy.py` (dev only)
- Eval + gate: `scripts/eval_rerank_abstain_policy.py`

## Problem it targets

The v5 reranker lifts where there is headroom (medium+hard: WebFAQ +0.370, GermanQuAD +0.346,
DT-test +0.542) but **churns near-ceiling GermanQuAD** (overall −0.0285, no_room 0.84, catastrophic
0.169). The fix: don't rerank lists whose first stage is already confident.

## Inference uses observable features ONLY

`extract_features` / `apply_policy` read **only** `first_stage_score`, `first_stage_rank`,
`reranker_score`, `candidate_source`, `doc_id`. They **never** read qrels / `positive_doc_ids` /
`label` / `teacher_score` / `oracle_ndcg` / `hardness_bucket`. Those appear only in fitting (dev
labels) and eval/analysis. (Unit-tested: stripping all labels from a row does not change the
policy's output.)

### Features (15)
first_stage_top1_score, first_stage_top2_score, first_stage_top1_top2_gap,
first_stage_top1_top5_gap, first_stage_score_entropy, candidate_source_agreement,
bm25_dense_agreement (None if only one source), reranker_top1_score, reranker_top2_score,
reranker_top1_top2_gap, reranker_score_entropy, max_rank_displacement,
reranker_rank_of_first_stage_top1, num_candidates, num_candidate_sources.

## Policies

`always_rerank`, `never_rerank`, `first_stage_confidence_abstain` (abstain if first-stage top1↔top2
gap ≥ threshold), `reranker_confidence_gate` (rerank only if reranker top1↔top2 gap ≥ threshold),
`displacement_guard` (don't let the reranker move the first-stage top1 below a max rank),
`conservative_blend` (`alpha·first_stage_norm + (1−alpha)·reranker_norm`; alpha=1 preserves the
first-stage order), and `combined_policy` (rerank **only if** first-stage confidence is low **and**
reranker confidence is high, with a displacement guard; otherwise keep/blend). Actions:
`keep_first_stage`, `rerank_top_k`, `conservative_blend`, `rerank_only_if_margin`.

## Fitting (dev only)

`fit_rerank_abstain_policy.py` grid-searches thresholds on a **dev split only** (WebFAQ dev or a
private/local dev). The grid is data-adaptive — quantiles of the dev gap features — plus blend
alphas and max-displacement ranks. **GermanQuAD/DT-test are never passed to the fitter**; they are
guardrails only. Objective: maximize dev policy nDCG@10 (tie-break to more abstention, then lower
catastrophic).

```bash
python scripts/fit_rerank_abstain_policy.py \
  --dev-lists <webfaq_dev_scored>.jsonl \
  --output outputs/v5-small-rag/abstain/fit_report.json \
  --markdown outputs/v5-small-rag/abstain/fit_report.md
```

## Evaluation + gate

```bash
python scripts/eval_rerank_abstain_policy.py \
  --policy outputs/v5-small-rag/abstain/fit_report.json \
  --eval-lists webfaq=<webfaq_test>.jsonl germanquad=<gq>.jsonl dt_test=<dt>.jsonl \
  --out-dir outputs/v5-small-rag/abstain
```

Per eval set: first_stage / always_rerank / policy nDCG@10, delta vs first-stage, delta vs
always_rerank, abstain_rate, rerank_rate, catastrophic_drop_rate, and metrics by hardness_bucket
(no_room/easy/medium/hard/impossible — oracle used for bucketing only, in analysis).

### Promotion gate for the policy
- WebFAQ overall Δ ≥ +0.05 and medium+hard Δ ≥ +0.20
- GermanQuAD overall Δ ≥ −0.005 and catastrophic ≤ 0.03
- DT-test overall Δ ≥ −0.005 and catastrophic ≤ 0.02
- the policy must **beat always_rerank** on GermanQuAD
- no qrels/hardness labels used at inference

## Acceptance

- A production-feasible abstention policy exists (features-only inference).
- It is fit without touching public guardrail labels (dev-only).
- It directly targets the observed v5 failure — near-ceiling GermanQuAD churn — via first-stage
  confidence abstention + a displacement guard.
