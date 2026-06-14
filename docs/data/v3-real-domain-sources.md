# v3 real-domain source acquisition

v2 proved you **cannot synthesize your way into admin/FAQ/legal generalization** — the teacher
rejected templated queries over Wikipedia (admin 4.8%, faq 5.7% validated). v3 sources **real,
licensed** German domain corpora. This framework is **fail-closed**: nothing trains until its
license is verified by a human.

Manifest: `configs/data_sources_v3.json` · loader/validator: `boldt_embed.domain_source_acquisition`
· CLI: `scripts/acquire_v3_sources.py`.

## Source categories (domains)

`faq_real`, `admin_real`, `legal_adjacency_real_no_eval_overlap` (the **real** domains v3 must
fill), plus `web`, `wiki_non_eval`, `german_stress`, `cross_lingual_de_en`.

## Input modes (`source_type`)

| mode | meaning | materialized by |
|---|---|---|
| `local_jsonl` | user-provided query→document (or text) rows | `materialize-local` |
| `local_corpus_jsonl` | documents only; query generation later is marked `generated` | `materialize-local` |
| `hf_dataset` | optional HF dataset | `download-hf` only (never in dry-run/tests) |
| `url_manifest` | metadata only — **no scraping** unless implemented later | never |

## Manifest fields

```json
{
  "source_id": "...", "display_name": "...", "domain": "...",
  "source_type": "local_jsonl|local_corpus_jsonl|hf_dataset|url_manifest",
  "license": "...", "license_url": "...",
  "license_verified": true,
  "allowed_for_training": true,
  "eval_only": false, "public_benchmark": false,
  "contains_eval_overlap_risk": false,
  "requires_attribution": false,
  "supplemental": false,
  "notes": "...", "loader": {...}, "expected_fields": {...}
}
```

## Fail-closed rules (validation errors → manifest is rejected)

A source may set `allowed_for_training=true` **only if all** of:
- `license_verified=true` **and** the license string is concrete (not `unknown`/`verify`/`tbd`/`?`/empty);
- `public_benchmark=false` — **a public benchmark can never be a training source**;
- `eval_only=false`;
- `contains_eval_overlap_risk=false` — run the full leakage scan (`scripts/run_full_leakage_scan.py`)
  and have a human clear the flag first.

So: **if a license is uncertain, `allowed_for_training` must be false** until a human verifies it
and flips both flags. `license_verified=true` with an uncertain license string is also rejected.

## Synthetic data is supplemental only

A synthetic/generated source must set `supplemental=true`. Supplemental sources may be used as
extra training data but are **not counted toward the `*_real` coverage targets**
(`real_domain_coverage` / `real_domains_missing` in the summary count non-supplemental sources
only). This enforces the v3 goal: real domains need *real* data.

## Adding real admin / FAQ / legal data (the safe path)

1. Drop your rows (see schema below) at the manifest's loader path, e.g.
   `data/raw/v3/admin_real.jsonl`, `data/raw/v3/faq_real.jsonl`,
   `data/raw/v3/legal_adjacency_real.jsonl`.
2. Set the **real** `license` + `license_url` on the source entry. **Verify it yourself**, then
   set `license_verified: true`.
3. For legal-adjacent / Wikipedia-derived data: run the full leakage scan vs the eval corpora,
   then set `contains_eval_overlap_risk: false`.
4. Set `allowed_for_training: true`. The manifest now validates and the source materializes.

Local row schema (`validate_local_jsonl_row`): **`id`** (or `doc_id`/`query_id`), **`text`** or
(**`query`** and **`document`**), **`source_id`**, **`license`**; `url`/`title` optional.

## CLI

```bash
# Validate + plan (no network, no corpus I/O):
python scripts/acquire_v3_sources.py --manifest configs/data_sources_v3.json \
  --output-dir data/raw/v3 --mode dry-run

# Materialize local drops (reads + validates rows, writes data/raw/v3/<source_id>.jsonl):
python scripts/acquire_v3_sources.py --mode materialize-local --output-dir data/raw/v3

# Strict gate: fail if ANY source is still license_verified=false (use before a release run).
python scripts/acquire_v3_sources.py --mode dry-run --fail-on-unverified-license
```

`--fail-on-unverified-license` exits non-zero while unverified placeholders remain — that is the
gate working. The acquisition summary reports `rows_by_source/domain/license`, `materialized`,
`blocked` (with reasons), `real_domain_coverage`, `real_domains_missing`, `supplemental_sources`,
and `unverified_sources`.

## Current status

`admin_real` / `faq_real` / `legal_adjacency_real_no_eval_overlap` are **blocked placeholders**
(unverified license, `allowed_for_training=false`). `mmarco_de` and `clips_mqa_de` stay
**blocked/future** until their licenses + script-free loaders are verified. `dt_de_dpr`
(wiki_non_eval) is license-verified but blocked until the leakage scan clears its overlap risk.
Only `ger_backtrans_paraphrase` (web) and the supplemental synthetic stress set are currently
trainable — so `real_domains_missing` is all three, by design, until real data is dropped in.
