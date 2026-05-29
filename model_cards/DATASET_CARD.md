---
language: de
license: cc-by-sa-4.0
task_categories: [sentence-similarity, text-retrieval]
tags: [german, embeddings, contrastive, synthetic]
pretty_name: Boldt-Embed-DE training & evaluation data
---

# Dataset Card — Boldt-Embed-DE training & evaluation data

Contrastive German query/document pairs and triples (with hard negatives) plus small
evaluation sets. **Status: toy/sample data shipped; production corpora pending license
clearance** (see `docs/data/data-sources.md`).

## Composition
- `data/samples/toy_triples_de.jsonl` — 7 triples, all 7 hard-negative families (`synthetic`).
- `data/samples/toy_pairs_de.jsonl` — 6 pairs (`cc-by-sa-4.0`).
- `benchmarks/` — toy retrieval, STS, classification, clustering, cross-lingual DE→EN,
  RAG, and scored stress sets (plumbing-scale).
- Schema: `schemas/training_pair.schema.json` (query, positive, negatives, neg_types,
  query_type, source, license, lang).

## Intended use
First-stage retrieval / similarity / reranking training and *plumbing* evaluation for the
Boldt-Embed-DE family. Sample data is for pipeline validation, **not** model-quality claims.

## Limitations
- Tiny scale (single/low double digits per set) — not statistically meaningful for quality.
- Synthetic and template-derived examples may not reflect real query distributions.
- Cross-lingual and clustering sets are minimal.

## Evaluation
These sets are consumed by `scripts/run_eval_suite.py` and `run_local_benchmark.py`. Public
held-out evaluation uses MMTEB German + GermanDPR/GermanQuAD (ADR-005), not this sample data.

## License
- Sample/synthetic data: `synthetic` and `cc-by-sa-4.0` (per record `license`). Only the
  allowlist in `docs/data/license-policy.md` is accepted. Attribution/share-alike preserved.
- A published dataset derived from share-alike sources inherits SA terms.

## Reproducibility
- Synthetic generation: `data/synthetic/prompt_specs.json` (versioned) + deterministic
  `boldt_embed.hard_negatives` generators.
- Validation: `scripts/validate_data_schema.py` (schema + license + PII + leakage).
- PII (`data.scan_pii`) and benchmark leakage (`data.find_leakage`) are checked; see
  `docs/data/leakage-policy.md`.
