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

### Verified concrete training sources (2026-05-29 — see `training-datasets-research-2026.md`)
Permissive, **non-benchmark** German sources, disjoint from the eval set:

| HF id | License | Domain | Notes |
|---|---|---|---|
| `unicamp-dl/mmarco` (`german`) | Apache-2.0 (card; verify MS-MARCO upstream) | web | top pick; non-Wikipedia → disjoint from GermanQuAD/GerDaLIR |
| `clips/mqa` (`de`) | CC0-1.0 | web FAQ/CQA | cleanest license |
| `deutsche-telekom/wikipedia-22-12-de-dpr` | CC-BY-SA-4.0 (code MIT) | Wikipedia 2022-12 | German-native DPR + formal/informal imperative variants; questions not from GermanQuAD; ⚠️ dedup vs GermanQuAD/MIRACL |
| `nthakur/swim-ir-monolingual` (`de`) | CC-BY-SA-4.0 | Wikipedia | 447k synthetic; ⚠️ dedup vs GermanQuAD/MIRACL |
| German Wikipedia / FineWeb-2 (de) | CC-BY-SA-4.0 / ODC-By | wiki / web | for synthetic + mined pairs |

The earlier GermanQuAD-train→GermanQuAD-test run is **in-domain** and is superseded by training
on the web sources above and evaluating on the held-out Wikipedia/legal benchmarks.

## Rules
- Only the licenses in `ALLOWED_LICENSES` (`license-policy.md`).
- GermanQuAD/GermanDPR/GerDaLIR/MIRACL/MLDR/STS22/MASSIVE/PawsX/XNLI are **eval-only** — never
  train on them (`leakage-policy.md`).
- Every batch passes `validate_dataset` + `check_licenses` + `scan_pii` + `find_leakage`.
- Concrete URLs + license + retrieval date are recorded here per source before use.

## Toy data shipped now
- `data/samples/toy_triples_de.jsonl` (7 triples, all six core hard-neg families, `synthetic`).
- `data/samples/toy_pairs_de.jsonl` (6 pairs, `cc-by-sa-4.0`).
