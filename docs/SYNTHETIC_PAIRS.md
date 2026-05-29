# Synthetic Pair Generation

Implements the synthetic-data parts of ADR-004. Goal: reproducible, license-clean,
German contrastive pairs with strong, category-tagged hard negatives.

## Workflow
1. **Source passages** from license-compatible German corpora (see DATA_PLAN).
2. **Generate a query** per passage (`query_from_passage` template); optionally a
   `paraphrase_positive`.
3. **Generate/derive hard negatives** per German family. Two paths:
   - **LLM templates** (`prompt_specs.json`) for natural negatives.
   - **Rule-based** `boldt_embed.hard_negatives.make_hard_negatives()` for deterministic,
     reproducible negatives across the six families.
4. **Filter** every pair with `boldt_embed.hard_negatives.filter_pair()` — rejects
   non-German text, empty/over-long fields, `query == positive`, negatives equal to the
   positive, and negatives that are near-duplicates of the positive (false negatives).
5. **Leakage check** the batch with `data.find_leakage()` against the eval registry.
6. **Record provenance**: generator model id, decoding params, prompt `version`, and the
   filter settings used — stored with each produced batch.

## Reproducibility
- Rule-based negatives are fully deterministic (no RNG).
- LLM-based generation is reproduced from: `prompt_specs.json` version + generator model id
  + decoding params. Bump the `version` field whenever a template changes.

## Hard-negative families (7)
`compound`, `negation`, `legal_ref`, `dates_numbers`, `regional_variant`,
`entity_disambiguation`, `outcome_flip` (same facts, opposite result). Each shipped toy triple
exercises at least one; `tests/test_hard_negatives.py` + `test_query_types.py` check generators.

## Query types (10)
`keyword`, `question`, `short_vague`, `long_detailed`, `entity`, `date_number`, `legal_admin`,
`support`, `summary`, `negation_contradiction` — enumerated in `boldt_embed.hard_negatives.QUERY_TYPES`
with one LLM template each in `data/synthetic/prompt_specs.json -> query_type_templates`. The
`query_type` field is part of the training-pair schema.
