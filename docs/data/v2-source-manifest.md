# v2 data-source manifest

`configs/data_sources_v2.json` is the **auditable gate** for what may enter v2 training. It is
validated by `src/boldt_embed/source_manifest.py` and `scripts/validate_data_sources_v2.py`,
and it **fails closed**: a source is training-eligible only if it clears every rule below.

## Entry schema

```json
{
  "source_id": "...", "display_name": "...",
  "source_type": "local_jsonl|hf_dataset|synthetic|derived",
  "domain": "web|faq|admin|legal_adjacency_no_eval_overlap|wiki_non_eval|german_stress|cross_lingual_de_en  (training)  |  qa_wiki|legal|sts|clustering (eval-only)",
  "license": "...", "license_url": "(optional)",
  "allowed_for_training": true/false,
  "public_benchmark": true/false,
  "eval_only": true/false,
  "notes": "...",
  "loader": {"kind": "jsonl|hf", "path_or_id": "...", "config": "(opt)", "split": "(opt)"},
  "expected_fields": {"query": "...", "document": "...", "title": "...", "positive": "..."}
}
```

## Fail-closed rules (validation errors)

A source is rejected (and `validate_data_sources_v2.py` exits non-zero) if:
- **license is missing/empty**;
- `allowed_for_training=true` **and** `public_benchmark=true` (public benchmarks never train);
- `allowed_for_training=true` **and** `eval_only=true`;
- `allowed_for_training=true` **and** the license string is **uncertain** (contains
  `uncertain`/`unknown`/`verify`/`tbd`/`?`);
- `allowed_for_training=true` **and** the domain is not one of the seven **training** domains;
- unknown `domain`, bad `source_type`/`loader`, or duplicate `source_id`.

Net effect: **if a license is uncertain, `allowed_for_training` must be false.**

## Provenance carried onto every candidate (and into the teacher cache)

The manifest is the **single source of truth** for license provenance. When a candidate row is
built, `source_manifest.candidate_provenance(entry, row)` stamps it with:

| field | meaning |
|---|---|
| `source_id` / `source` | the manifest entry the row was admitted under |
| `license` | concrete license — for `…-inherits-source` sources, the seed passage's real license (e.g. `CC-BY-SA-4.0`); else the entry's license |
| `license_url` | optional, when known |
| `license_origin` | `manifest` (concrete license on the source) or `inherited` (synthetic data inheriting its seed's license) |
| `inherited_from_source_id` | for `inherited` rows, the seed source (when recorded) |
| `allowed_for_training` | the entry's training-permission bit, per row |

These fields are then carried **verbatim** through `build_v2_candidates.py` →
`build_teacher_cache.py` (`make_cache_row`) → `summarize_teacher_cache.py`. This closes the v2
bug where the teacher-cache summary reported `by_license {"unknown": N}` despite the manifest
having real licenses (the license was being dropped when the cache row was built). The cache
summary now reports `unknown_license_rows` / `disallowed_for_training_rows`, and both the
summarizer (`--fail-on-unknown-license`, `--fail-on-disallowed-training-source`) and the release
gate (`validate_release_2026.py --require-v2-artifacts|--require-v3-artifacts`) **fail** if
either is > 0. See `docs/teacher-cache.md`.

## How to add new data safely

1. Add an entry with the real `license` (and `license_url`). If you're not sure of the
   license, set `allowed_for_training: false` until you've verified it.
2. Use a **training** domain for training sources; eval/benchmark sources use an eval domain
   and must be `eval_only: true`, `public_benchmark: true`, `allowed_for_training: false`.
3. Run `python scripts/validate_data_sources_v2.py --manifest configs/data_sources_v2.json`.
4. Only then does the v2 candidate builder (`build_v2_candidates.py`) pull from it.

## Current manifest

Six training-allowed sources (`local_web_v2`, `local_admin_v2`, `dt_de_dpr`,
`ger_backtrans_paraphrase`, `swim_ir_de`, `synthetic_adversarial`). Blocked: `mmarco_de` and
`clips_mqa_de` (uncertain license / loader needs a deprecated script); `germanquad`, `gerdalir`,
`miracl_de`, `mldr_de`, `sts22_de` (public benchmarks → **eval-only**). `dt_de_dpr`/`swim_ir_de`
carry an explicit Wikipedia-overlap warning: dedup + leakage-filter against GermanQuAD/MIRACL
before training. Tests never download external datasets — only the manifest is validated.
