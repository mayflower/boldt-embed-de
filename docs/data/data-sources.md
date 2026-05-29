# Data Sources

German training data for Boldt-Embed-DE. Status: **policy + toy data**; real corpora are
added to the table as their licenses clear (ADR-004). Each shipped record carries `source`
and `license`; the gate is `scripts/validate_data_schema.py`.

## Dataset categories (prompt 04)

| Category | Example source type | License posture | Status |
|---|---|---|---|
| General German web passages | FineWeb-2 (de) filtered | ODC-By / per-source | candidate |
| Wikipedia / public reference | dewiki, Wikinews | CC-BY-SA-4.0 | candidate |
| Legal/admin text | public-domain statutes, gov FAQ | public-domain / permissive | candidate |
| Technical documentation | permissively-licensed docs | per-source (must be permissive) | candidate |
| Product/support documentation | support FAQs (where licensed) | per-source | candidate |
| German QA data | GermanQuAD (train only) | CC-BY-4.0 | candidate (test = eval-only) |
| Parallel DE-EN | OPUS subsets (permissive) | per-source | candidate (cross-lingual) |
| Synthetic pairs | generated (prompt_specs.json) | `synthetic` | shipped (toy) |

## Rules
- Only the licenses in `ALLOWED_LICENSES` (`license-policy.md`).
- GermanQuAD/GermanDPR **test** splits are eval-only — never train on them (`leakage-policy.md`).
- Every batch passes `validate_dataset` + `check_licenses` + `scan_pii` + `find_leakage`.
- Concrete URLs + license + retrieval date are recorded here per source before use.

## Toy data shipped now
- `data/samples/toy_triples_de.jsonl` (7 triples, all six core hard-neg families, `synthetic`).
- `data/samples/toy_pairs_de.jsonl` (6 pairs, `cc-by-sa-4.0`).
