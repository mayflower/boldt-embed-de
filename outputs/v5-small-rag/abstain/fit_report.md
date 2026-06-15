# rerank-or-abstain fit (combined_policy)

Fit on DEV ONLY: `outputs/v5-small-rag/eval/webfaq_dev_scored.jsonl` (660 lists). Guardrails not used.

## Best params

```json
{
  "fs_gap_threshold": 15.068491,
  "rr_gap_threshold": 0.28125,
  "alpha": 0.5,
  "max_displacement_rank": 10
}
```

## Dev metrics (best policy)

- dev nDCG@10: 0.74022 (first-stage 0.61317, always_rerank 0.772447)
- delta vs first-stage: +0.12705
- delta vs always_rerank: -0.032227
- abstain_rate: 0.2  rerank_rate: 0.8
- catastrophic_drop_rate: 0.007576

_grid: fs_gaps=[0.481459, 1.602395, 3.262757, 5.017926, 6.925121, 10.314034, 15.068491], rr_gaps=[0.28125, 0.844727, 1.552734, 2.046875, 2.78125, 4.28125, 6.425781], alphas=[1.0, 0.7, 0.5], max_disp=[3, 5, 10]; 441 trials_
