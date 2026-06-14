# Teacher-threshold calibration (v3)

v2 filtered every teacher-scored positive with a **single** reranker threshold (`>= 2.0`) and
fed the same set to both students. The reranker then trained on noisy positives (in v2 the
positive teacher-median was ≤ the negative median) and degraded GermanQuAD. v3 calibrates the
threshold per consumer so the **embedder and reranker no longer share one noisy positive set**.

Module: `boldt_embed.teacher_calibration` · CLI: `scripts/calibrate_teacher_thresholds.py` ·
outputs: `qwen3_v3.calibration.{json,md}`, `qwen3_v3.filtered_embedder.jsonl`,
`qwen3_v3.filtered_reranker.jsonl`.

## Compatibility with v2 scoring

`scripts/build_teacher_cache.py` is unchanged — it still scores (query, document) pairs with the
Qwen3 teachers and writes rows carrying `reranker_score`/`embedding_score` plus full provenance
(`domain`/`source`/`license`/`license_origin`/`allowed_for_training`). Calibration is a **new
post-processing step** over that cache; the old `summarize_teacher_cache.py --reranker-threshold`
path still works (and now also accepts `--by-threshold` to print the sweep).

## What it reports

- **Acceptance by threshold** at `-2, 0, 1, 2, 3, 4, 5`, overall and **by domain / source /
  license** — so threshold sensitivity is explicit (e.g. how many admin positives survive at 2 vs 4).
- **Median reranker score** of the embedder-accepted vs reranker-accepted sets.
- **Low-score positives** — positives the teacher scored below threshold (suspicious / mislabeled).
- **High-score rejected** — negatives the teacher scored high (likely false negatives).

## Domain-aware thresholds

- Global default: **embedder `>= 2.0`**, **reranker `>= 4.0`** (stricter → higher precision).
- Per-domain overrides are **allowed but not required** — set them in the v3 config under
  `teacher_calibration.per_domain_embedder` / `per_domain_reranker`.

```json
"teacher_calibration": {
  "embedder_threshold": 2.0,
  "reranker_threshold": 4.0,
  "per_domain_reranker": {"faq_real": 4.5, "admin_real": 4.5},
  "max_suspicious_positive_rate": 0.5
}
```

The reranker set is a **strict, higher-precision subset** of the embedder set: every
reranker-kept positive clears the stricter threshold, while the embedder set retains the
2.0–4.0 band.

## Outputs

- `qwen3_v3.filtered_embedder.jsonl` — positives `>= embedder threshold` (per-domain aware).
- `qwen3_v3.filtered_reranker.jsonl` — positives `>= reranker threshold` (stricter).
- `qwen3_v3.calibration.json` / `.md` — the report above + gate results.

## Gates (failure → non-zero exit → training blocked)

- `license_unknown_rows_zero` — zero unknown/missing-license rows.
- `real_domain_min_accepted` — each real domain's **embedder-accepted** count ≥ the config floor
  (`domain_quality_gates.min_real_domain_accepted`).
- `suspicious_positive_rate` — fraction of positives the teacher scored below the embedder
  threshold ≤ `max_suspicious_positive_rate` (default 0.5).

## CLI

```bash
python scripts/calibrate_teacher_thresholds.py \
  --teacher-cache outputs/v3-real-domain/teacher-cache/qwen3_v3.manifest.json \
  --config configs/experiments/v3_real_domain_generalization.json \
  --output  outputs/v3-real-domain/teacher-cache/qwen3_v3.calibration.json \
  --markdown outputs/v3-real-domain/teacher-cache/qwen3_v3.calibration.md \
  --embedder-output outputs/v3-real-domain/teacher-cache/qwen3_v3.filtered_embedder.jsonl \
  --reranker-output outputs/v3-real-domain/teacher-cache/qwen3_v3.filtered_reranker.jsonl
# exit 0 = gates pass; exit 1 = a gate failed -> do not train.
```

Train the embedder on `filtered_embedder.jsonl` and the reranker candidate lists on
`filtered_reranker.jsonl` — see `docs/hard-negative-mining-2026.md` and
`docs/reranker-training-2026.md`.
