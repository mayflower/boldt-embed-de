---
language: de
license: apache-2.0
library_name: sentence-transformers
pipeline_tag: feature-extraction
base_model: Boldt/Boldt-DC-350M
tags: [german, embeddings, retrieval, matryoshka, causal]
---

# Boldt-Embed-DE-350M-v1-causal

German-first text embedding model: a **causal decoder** embedder built on
`Boldt/Boldt-DC-350M` with **EOS/last-token pooling** and Matryoshka-truncatable vectors.

> **Status: untrained scaffold.** Architecture, instruction format, pooling, and the
> evaluation harness are implemented and validated, but the contrastive training run has
> not been executed in this repository. No quality numbers are claimed below.

## Intended use
- Asymmetric German retrieval (query → document), semantic similarity, clustering.
- Prepend the query instruction to queries; pass documents with no/!light template.
- Embeddings are L2-normalized; Matryoshka prefixes (1024→64) must be **re-normalized**.

## Usage
```python
from sentence_transformers import SentenceTransformer  # requires the trained model
model = SentenceTransformer("Boldt/Boldt-Embed-DE-350M-v1-causal")
q_instr = ("Instruct: Repräsentiere die Suchanfrage für die Suche nach passenden "
           "deutschen Dokumenten.\nQuery: ")
queries = [q_instr + "Wie hoch darf die Mietkaution sein?"]
docs = ["Die Mietkaution darf höchstens das Dreifache der Nettokaltmiete betragen."]
q = model.encode(queries, normalize_embeddings=True)
d = model.encode(docs, normalize_embeddings=True)
# Matryoshka: take q[:, :256] and d[:, :256], then re-normalize before cosine.
```

## Training
- Base: `Boldt/Boldt-DC-350M` (German base LM, apache-2.0).
- Objective: MultipleNegativesRanking / InfoNCE with German hard negatives.
- Config: `configs/training_causal.json`. Dry-run: `python scripts/train_causal.py --dry-run`.
- Data: license-clean German pairs/triples + synthetic (see DATA_PLAN, ADR-004).

## Evaluation
**Pending.** To be populated from a saved MTEB run (`scripts/run_mteb_benchmark_template.py`)
with full run metadata, per ADR-005 and `docs/BENCHMARK_PLAN.md`. The repo's local benchmark
validates plumbing only and is **not** a quality claim. No benchmark numbers are reported here
until a real run exists under `outputs/`.

## Limitations
- 350M-class, German-first: not a "best multilingual" model.
- **No long-context claim** (8k/32k) without a trained+evaluated context-extension phase.
- Native 1024-d output assumes the base hidden size ≥ 1024 — **MUST-VERIFY** against the base
  `config.json` (ADR-003); a projection head may be required.
- Last-token pooling can under-weight early-sequence content vs. bidirectional pooling.
- Not instruction/chat tuned; the "instruction" is a representation prompt.

## License
- **Code:** Apache-2.0.
- **Base weights:** `Boldt/Boldt-DC-350M` is apache-2.0 (verified 2026-05-28).
- **Derivative weights:** intended apache-2.0, contingent on every training dataset's license
  (ADR-004). Confirm before publishing weights.

## Reproducibility
- Base model, config, instruction format, and pooling are pinned above and in `configs/`.
- Validate the pipeline: `make all`. Dry-run the trainer: `make dry-run-causal`.
- Record commit + run metadata with any evaluation (ADR-005).
