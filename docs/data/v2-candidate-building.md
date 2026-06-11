# v2 candidate building

`scripts/build_v2_candidates.py` turns manifest-approved sources into the v2 training-candidate
set at scale (50k–250k), with the same fail-closed safety the manifest enforces, plus dedup,
PII, leakage, and domain balancing. Pure stdlib, streaming, no network, no ML.

## Pipeline

1. **Admit (manifest-gated):** each input row names a `source` (a `source_id` in
   `configs/data_sources_v2.json`). Rows from unknown sources or sources without
   `allowed_for_training=true` are **dropped** (counted as `blocked_unknown_source` /
   `blocked_not_allowed_for_training`). License comes from the manifest, not the row.
2. **Dedup** (`--dedup`): collapse identical (query, document) pairs by `pair_hash`.
3. **PII scan** (`--pii-scan`): `data.scan_pii` over query+document. **Fails the run** if any
   PII is found, unless `--allow-pii` (which drops the offending rows instead). Writes a
   by-kind summary.
4. **Leakage** (`--leakage-corpus-jsonl`): `data.find_leakage` (exact + token-Jaccard ≥
   `--leakage-threshold`) against eval corpora; leaked rows dropped.
5. **Domain balance:** with `--domain-config configs/experiments/v2_generalization.json`,
   sample to the per-domain target counts (`target_fraction × --target-count`), deterministic
   by `--seed`. Reports requested vs actual per domain.

Output rows add `text_hash`, `pair_hash`, and `metadata.source_id` to the standard candidate
schema. A `<output>.report.json` records admit/blocked counts and final domain distribution.

## Run it

```bash
# Dry-run on tiny fixtures (no write):
python scripts/build_v2_candidates.py \
  --manifest tests/fixtures/v2_sources_manifest.json \
  --source-jsonl tests/fixtures/v2_candidates_seed.jsonl \
  --domain-config configs/experiments/v2_generalization.json \
  --target-count 20 --dedup --pii-scan --dry-run

# Real build (streams large files):
python scripts/build_v2_candidates.py \
  --manifest configs/data_sources_v2.json \
  --source-jsonl data/raw/v2/*.jsonl \
  --output data/processed/candidates_v2.jsonl \
  --target-count 50000 --dedup --pii-scan \
  --leakage-corpus-jsonl data/processed/eval_leakage.jsonl
```

## Safety notes

- A row can only enter training if its source is `allowed_for_training` in the manifest — there
  is no per-row license override (prevents accidentally training on uncertain-license data).
- PII defaults to **failing**; do not pass `--allow-pii` on real corpora without reviewing the
  summary. Leakage filtering is mandatory whenever Wikipedia-derived sources (DT-de-dpr,
  swim-ir) are used, since they overlap GermanQuAD/MIRACL.
- No generated `data/processed/*.jsonl` is committed (git-ignored); only tiny fixtures are
  tracked, and tests never download external datasets.
