# v2 results template

`scripts/summarize_v2_results.py` generates `V2_RESULTS.md` + `V2_RESULTS.json` — the
single artifact that decides whether the v2 data-scale run improved the project. Pure stdlib;
missing inputs are warned, not fatal.

```bash
python scripts/summarize_v2_results.py \
  --v1-dir outputs --v2-dir outputs/v2-generalization \
  --config configs/experiments/v2_generalization.json \
  --output outputs/v2-generalization/V2_RESULTS.md \
  --json-output outputs/v2-generalization/V2_RESULTS.json
```

## Report sections

1. **Executive verdict** — `improved` (all success criteria met **and** reranker gate passes),
   `failed` (dense GermanQuAD and DT-test both below min), else `mixed`.
2. **Dense retrieval** — best student nDCG@10 per dataset, v1 vs v2, against the config minima.
3. **OOD legal** — GerDaLIR (the v1 weak spot).
4. **Reranker lift** — DT-test / GermanQuAD deltas + the promotion-gate result.
5. **Matryoshka** — 256-d retention vs the ≥0.95 target.
6. **Recommendations** — auto-generated, one per failed criterion (what to scale/retrain).
7. **Warnings** — any missing input reports.

## Success criteria (from `configs/experiments/v2_generalization.json`)

dense GermanQuAD ≥ 0.88, DT-test ≥ 0.95, GerDaLIR ≥ 0.10 (stretch 0.12); reranker GermanQuAD
delta ≥ 0.0 (target +0.02); Matryoshka 256-d retention ≥ 0.95. The verdict and recommendations
are computed mechanically from these — no manual spreadsheet, and every number traces to a
saved report with a run card.
