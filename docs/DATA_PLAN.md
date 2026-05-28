# Data Plan (licensing, schema, leakage)

Implements ADR-004 (data & licensing) and ADR-005 (benchmark protocol / leakage).

## Record schema
Training data is JSONL, one contrastive record per line; spec in
`data/schema/pair_schema.json`, enforced in code by `boldt_embed.data.validate_record`.
Required: `query`, `positive`, `source`, `license`. Optional: `negatives`, `neg_types`,
`lang` (default `de`). See `data/samples/toy_triples_de.jsonl` and `toy_pairs_de.jsonl`.

## Sources & licensing
- **Accept only permissive licenses:** CC0, CC-BY-4.0, CC-BY-SA-4.0, Apache-2.0, MIT,
  public-domain, plus our own `synthetic` data. (`ALLOWED_LICENSES` in `data.py`.)
- Every record carries `source` + `license`; `check_licenses()` fails the build on any
  disallowed license. License of the base **weights** (apache-2.0, ADR-001) is tracked
  separately from each **dataset** license.

| Category | Example source type | License posture |
|---|---|---|
| Encyclopedic German | Wikipedia/Wikinews-style | CC-BY-SA-4.0 (attribution + share-alike) |
| Public admin / legal info | gov FAQ / public-domain statutes | public-domain / permissive |
| Synthetic pairs | generated (see synthetic-pair specs) | `synthetic` (generator versioned) |

> Concrete corpus URLs are added here as they are licensed-cleared; until then this is policy.

## Benchmark-leakage control (ADR-005)
- Maintain a **registry of evaluation corpora** (GermanDPR/GermanQuAD, MMTEB German tasks).
- `find_leakage(records, eval_texts)` flags **exact** (normalized) and **near-duplicate**
  (token Jaccard ≥ 0.9) overlap between training `query`/`positive` and any eval text.
- No public **test** split ever enters training or train-time validation.
- Train-time validation uses a **private dev split**, disjoint from all public test data.

## German hard negatives
Negatives are tagged with `neg_types` drawn from: `compound`, `negation`, `legal_ref`,
`dates_numbers`, `regional_variant`, `entity_disambiguation` (+ `lexical`, `random`).
Generators live in `boldt_embed.hard_negatives`; the toy triples exercise each family.

## Validation gates (data phase)
1. `validate_dataset()` returns zero errors on shipped samples.
2. `check_licenses()` returns no disallowed licenses.
3. `find_leakage()` against the eval registry returns no un-waived hits.
4. Synthetic generation prompt + filters are versioned (see synthetic-pair specs).

These are exercised by `tests/test_data.py` and reported by the smoke suite.
