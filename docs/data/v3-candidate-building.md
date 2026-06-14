# v3 candidate building

Builds the v3 training-candidate set from **real domain corpora** with strict per-domain quotas
and full provenance. Replaces `build_v2_candidates.py` for the next experiment.

Script: `scripts/build_v3_candidates.py` · row builders: `boldt_embed.data_pipeline`
(`build_v3_candidate_row`, `build_v3_passage_record`, `quota_report`).

## Inputs

- `configs/data_sources_v3.json` — the fail-closed source manifest (only training-allowed,
  license-verified sources are admitted).
- `configs/experiments/v3_real_domain_generalization.json` — `domain_targets` (fractions) →
  per-domain quotas = fraction × `--target-count`.
- `data/raw/v3/<source_id>.jsonl` — materialized drops (from `acquire_v3_sources.py`).
- the eval leakage index (`scripts/build_leakage_index.py` / `run_full_leakage_scan.py`).

## Output row schema (every pair carries provenance end-to-end)

```json
{
  "record_type": "pair",
  "query_id": "...", "doc_id": "...", "query": "...", "document": "...", "positive": true,
  "source_id": "...", "source": "...", "domain": "...",
  "license": "...", "license_origin": "manifest|inherited",
  "allowed_for_training": true, "synthetic": false,
  "source_url": "(optional)", "text_hash": "...", "pair_hash": "..."
}
```

## Document-only corpora

`local_corpus_jsonl` sources emit **passage records** (`record_type: "passage"`, no query) to
`passages_v3.jsonl` — the builder **never fabricates a query/doc pair**. A later generator
produces synthetic pairs and marks them `synthetic: true`; until then a document-only domain's
*pair* quota is legitimately **missed** (passages ≠ pairs).

## Quotas

Per-domain targets come from the v3 config. For the **real** domains (`faq_real`, `admin_real`,
`legal_adjacency_real_no_eval_overlap`) only **non-synthetic** pairs count toward the target — a
synthetic source cannot satisfy a real-domain quota (the v2 lesson). The report gives, per
domain, `target / total / real / synthetic / counted_toward_target / achieved` plus a `missed`
list. `--fail-on-domain-quota-miss` turns a miss into a non-zero exit.

## Safety

- **PII scan** on by default (`--pii-scan`; disable only with `--no-pii-scan`) — offending pairs
  are dropped.
- **Leakage** — candidates matching the eval leakage index are dropped. A leakage index is
  **required to materialize** candidates; it may be skipped **only in `--dry-run`** (which writes
  nothing), with a loud warning (`--allow-no-leakage-index` to acknowledge).
- **Unknown license** — rows with an unknown/missing license are dropped; `--fail-on-unknown-license`
  makes that a non-zero exit. (The manifest is already fail-closed: an allowed source must be
  license-verified, so unknown licenses should never reach here.)
- **Blocked sources** — anything not `allowed_for_training` (public benchmark, eval-only,
  unverified, overlap-risk) is skipped and listed in `blocked_sources` with the reason.

## Reports

`outputs/v3-real-domain/candidate_build_v3.{json,md}`: status, totals (selected pairs, passages,
real domains with real pairs), `dropped_by_reason`, `blocked_sources`, and the quota table
(achieved vs target, real/synthetic split).

## CLI

```bash
python scripts/build_v3_candidates.py \
  --manifest configs/data_sources_v3.json \
  --config configs/experiments/v3_real_domain_generalization.json \
  --raw-dir data/raw/v3 \
  --output outputs/v3-real-domain/candidates_v3.jsonl \
  --target-count 100000 \
  --leakage-index outputs/v3-real-domain/leakage/eval_index.json \
  --pii-scan --fail-on-unknown-license --fail-on-domain-quota-miss
```

## Why this exists

The build **proves real-domain coverage before teacher scoring**: with the shipped placeholder
manifest it correctly reports the `*_real` quotas as missed (no real data yet) and refuses (with
`--fail-on-domain-quota-miss`) to proceed — so we never spend 8B-teacher GPU hours on a set that
is secretly web+wiki. Provenance (license / origin / synthetic) rides every row through to the
teacher cache and the domain-quality gate. See `docs/data/v3-real-domain-sources.md`,
`docs/domain-quality-gates.md`, `docs/scalable-mining.md`.
