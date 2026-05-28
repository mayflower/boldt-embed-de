# ADR-004 — Training data and licensing

## Status
Accepted (2026-05-28) as policy; concrete source list is tracked in `docs/DATA_PLAN.md`.

## Context
Embedding quality depends on contrastive pairs/triples with strong hard negatives. Data
choices create the two biggest release risks: **license incompatibility** and **benchmark
leakage** (training on text that overlaps evaluation corpora).

## Decision
1. Use only **license-compatible** German sources (permissively licensed corpora + our own
   synthetic data). Record **per-source license, URL, and date** in the data manifest.
2. **Leakage control:** maintain a registry of evaluation corpora (GermanDPR/GermanQuAD,
   MMTEB German tasks) and **dedup/filter training data against them** (URL, doc-id, and
   near-duplicate text hashing). No public **test** split ever enters training or train-time
   validation.
3. **Synthetic data** is versioned: prompt template + generator model + filters are recorded,
   and synthetic pairs pass quality filters before use.
4. Train-time validation uses a **private dev split**, never public test labels.

## Consequences
- Every dataset must pass a `data.py` schema + license + leakage check before training.
- Adds a dedup/leakage step to the pipeline, but makes published benchmark numbers defensible.
- Synthetic generation is reproducible and auditable (see synthetic-pair specs).
