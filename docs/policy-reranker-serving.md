# Policy reranker — serving wrapper

A production-style wrapper that applies the bounded rerank **policy artifact** to already-scored
candidate lists. **Raw always-rerank is impossible unless explicitly requested.** Pure stdlib, no ML.

- Module: `src/boldt_embed/policy_reranker.py`
- CLI: `scripts/rerank_with_policy.py`
- Policy: `configs/policies/bounded_margin_override_v1.json` (loaded + validated via `policy_config`)

## Modes

| mode | behavior |
|---|---|
| `policy_gated` (default) | bounded margin_override — safe; top-k lock + bounded downshift/upshift + margin override + blend |
| `first_stage_only` | identity (keep first-stage order); every action `kept_first_stage` |
| `raw_rerank` | pure reranker order — **DISABLED** unless `--allow-raw-rerank-dangerous` (CLI) / `allow_raw=True` |

Loading the policy validates the artifact first, so a policy that (mis)claims raw-always-rerank is
recommended can't even be loaded.

## Inference safety

Policy decisions read only observable per-candidate fields — `first_stage_rank`, `first_stage_score`,
`reranker_score`, `candidate_source`, `doc_id` — and **never** qrels/labels/oracle/hardness/eval-set
(unit-tested: identical output whether or not those fields are present). Tie-breaks are deterministic
and input-order-independent.

## Input

```
{"query_id": "...", "query": "...", "candidates": [
  {"doc_id": "...", "text": "...", "first_stage_rank": 1, "first_stage_score": 0.123,
   "candidate_source": "dense|bm25|hybrid", "reranker_score": 0.42}]}
```
`reranker_score` is required for `policy_gated`/`raw_rerank`; if absent the wrapper keeps the first
stage. A malformed row (no `query_id`, empty `candidates`, a candidate missing `doc_id`, or a
candidate with neither `first_stage_rank` nor `first_stage_score`) raises a clear `ValueError`.

## Output

Per-candidate `final_rank`, `first_stage_rank`, `reranker_rank`, `final_score`, and a
`policy_action` ∈ {`locked`, `reranked`, `blended`, `margin_override`, `kept_first_stage`} with a
`policy_reason`, plus per-query `diagnostics` (`max_downshift`, `max_upshift`, `top_k_locked`,
`margin_override_used`, `num_candidates`).

## How `policy_gated` works (the bounds)

- Lock the first-stage **top-`preserve_first_stage_top_k`** (3) — action `locked` — unless the
  reranker beats the first-stage top-1 by `margin_override` (3.0), in which case that doc takes
  rank 1 (action `margin_override`).
- The tail is **blended** (`alpha·first_stage_norm + (1−alpha)·reranker_norm`, alpha 0.85 when the
  reranker agrees / 0.65 when the override fired) and ordered within bounds: no doc drops more than
  `max_downshift` (2) or rises more than `max_upshift_without_margin` (5) ranks unless its reranker
  score clears the margin.

## How to serve

```bash
python scripts/rerank_with_policy.py \
  --policy configs/policies/bounded_margin_override_v1.json \
  --input candidate_lists_scored.jsonl --output reranked_policy.jsonl
# raw always-rerank (unsafe) — must be explicit:
python scripts/rerank_with_policy.py --mode raw_rerank --allow-raw-rerank-dangerous ...
```

## Acceptance

- ✅ A safe serving path: `policy_gated` is the default; it bounds churn and reports diagnostics.
- ✅ Raw rerank requires an explicit dangerous flag (`--allow-raw-rerank-dangerous` / `allow_raw=True`);
  without it `raw_rerank` raises and the CLI exits non-zero.
