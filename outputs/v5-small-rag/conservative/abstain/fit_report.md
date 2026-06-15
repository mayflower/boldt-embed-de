# rerank-or-abstain fit (combined_policy)

Fit on DEV ONLY: `outputs/v5-small-rag/eval/conservative_scored/webfaq_dev.jsonl` (660 lists). Guardrails not used.

## Best params

```json
{
  "fs_gap_threshold": 15.068491,
  "rr_gap_threshold": 0.250977,
  "alpha": 0.5,
  "max_displacement_rank": 10
}
```

## Dev metrics (best policy)

- dev nDCG@10: 0.717365 (first-stage 0.61317, always_rerank 0.754265)
- delta vs first-stage: +0.104195
- delta vs always_rerank: -0.0369
- abstain_rate: 0.198485  rerank_rate: 0.801515
- catastrophic_drop_rate: 0.007576

_grid: fs_gaps=[0.481459, 1.602395, 3.262757, 5.017926, 6.925121, 10.314034, 15.068491], rr_gaps=[0.250977, 0.921875, 1.588867, 2.09375, 2.625, 3.863281, 5.730469], alphas=[1.0, 0.7, 0.5], max_disp=[3, 5, 10]; 441 trials_
