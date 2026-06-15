# Bounded reranking policies (inference-only)

Wrap the EXISTING (conservative) v5 reranker scores with bounded policies that cap how far the
reranker may disturb a confident first stage — directly targeting the residual catastrophic rank
churn on near-ceiling GermanQuAD lists. **No training, no Qwen, no rescoring.** Pure stdlib.

- Module: `src/boldt_embed/bounded_rerank.py`
- Fit (dev only): `scripts/fit_bounded_rerank_policy.py`
- Eval: `scripts/eval_bounded_rerank_policy.py`

## Inference uses observable features ONLY

`apply_policy` / `extract_features` read only `first_stage_rank` / `first_stage_score` /
`reranker_score` / `candidate_source` / `doc_id` + the observable features (first-stage gap/entropy,
reranker gap/entropy, source agreement, num candidates/sources, proposed max displacement). They
**never** read qrels / labels / oracle / hardness_bucket / eval-set name. Those are used only in fit
(dev labels) and eval/analysis. (Unit-tested: stripping labels does not change a policy's output.)

## Policies

`identity`, `always_rerank`, `top1_lock` (keep first-stage rank-1), `topk_lock` (keep first-stage
top-k relative order; k∈1,2,3,5,10), `bounded_downshift` (no doc drops > D ranks; D∈1,2,3,5,10),
`bounded_upshift` (no doc rises > U ranks unless reranker margin ≥ M), `margin_override` (break the
top-1 lock only when the reranker top beats the locked doc by ≥ M), `blend`
(`alpha·fs_norm + (1−alpha)·rr_norm`; alpha=1 = first-stage order), `confidence_conditional`
(high first-stage confidence → topk_lock, else rerank), `combined_safe_policy` (high→topk_lock+blend,
medium→bounded_downshift+margin_override, low→bounded_upshift). Deterministic tie-breaks throughout.

## Result (EXECUTED 2026-06-15, conservative-scored eval lists, no GPU)

Policies applied to the conservative reranker's scores. nDCG@10 delta vs first stage + catastrophic
drop rate (per-query ≤ −0.2). **GermanQuAD/DT-test used for analysis only — never for fitting.**

| policy | GermanQuAD Δ | GermanQuAD catastrophic | WebFAQ Δ | DT-test Δ |
|---|--:|--:|--:|--:|
| always_rerank (conservative) | +0.0087 | 0.123 | +0.136 | +0.021 |
| top1_lock | +0.0257 | **0.011** | +0.090 | +0.008 |
| topk_lock (k=3) | +0.0167 | **0.001** | +0.068 | +0.004 |
| bounded_downshift (D=2) | +0.0181 | 0.120 | +0.108 | +0.021 |
| **margin_override (M=3)** | **+0.0439** | **0.015** | +0.101 | +0.020 |

**The bounded policies directly fix the catastrophic churn:** `margin_override` takes GermanQuAD
catastrophic **0.123 → 0.015 (≤ 0.03)** and makes GermanQuAD overall the best of any approach
(**+0.0439**), while keeping WebFAQ (+0.10) and DT-test (+0.020) positive. `top1_lock`/`topk_lock`
clear the bar too (0.011 / 0.001).

### Honest limitation of dev-only fitting
The autofit (`fit_bounded_rerank_policy.py`, dev = WebFAQ) **selects `always_rerank`**, because
WebFAQ dev's own catastrophic rate is already low (0.009) — WebFAQ does not *exhibit* the
near-ceiling churn that GermanQuAD does, so a dev-nDCG objective has no signal to prefer a protective
policy. Automatically selecting `margin_override`/`top1_lock` **without** guardrail labels needs a
dev set that contains near-ceiling lists (e.g. a private near-ceiling web-QA dev), or a fit objective
that bounds displacement on the high-first-stage-confidence dev subset (observable). The policies are
proven to work; the *selection* is the open piece.

## CLI

```bash
python scripts/fit_bounded_rerank_policy.py --dev-lists <webfaq_dev_scored>.jsonl \
  --output outputs/v5-small-rag/bounded/fit_report.json --markdown outputs/v5-small-rag/bounded/fit_report.md --grid-search
python scripts/eval_bounded_rerank_policy.py --policy outputs/v5-small-rag/bounded/fit_report.json \
  --eval-lists webfaq=<wf>.jsonl germanquad=<gq>.jsonl dt_test=<dt>.jsonl --out-dir outputs/v5-small-rag/bounded
```

`--dry-run` imports no torch. Reports include policy_name, selected thresholds, abstain_rate,
lock_rate, avg max displacement, nDCG@10 before/after, delta, catastrophic_drop_rate, by-bucket
metrics, and per-query top catastrophic examples.

## Acceptance

- ✅ Evaluated without new GPU work (pure inference over existing scores).
- ✅ Directly target catastrophic rank churn (margin_override / top1_lock cut GermanQuAD catastrophic
  0.123 → 0.015 / 0.011, GermanQuAD overall positive).
- ✅ Production-feasible (observable inference features only).
- Open: dev-only fit on WebFAQ selects always_rerank; selecting the protective policy without
  guardrail labels needs a near-ceiling dev / displacement-bounded fit objective. Reranker remains
  **Experimental / not recommended** until that selection is closed and DT-test's marginal
  beats-always-rerank check is resolved.
