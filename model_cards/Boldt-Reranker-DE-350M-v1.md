---
language: de
license: apache-2.0
pipeline_tag: text-classification
base_model: Boldt/Boldt-DC-350M
tags: [german, reranker, cross-encoder, retrieval]
---

# Boldt-Reranker-DE-350M-v1

German **cross-encoder reranker** on `Boldt/Boldt-DC-350M`. Encodes (query, document)
together and emits a relevance score (or `Ja`/`Nein` logit). Used for production reranking,
hard-negative mining, and as a distillation teacher for the bi-encoders.

> **Status: untrained scaffold.** The cross-encoder input format, mining, and distillation
> helpers are implemented and validated; training has not been run here. No quality numbers
> are claimed below.

## Intended use
- Re-rank a candidate list retrieved by a first-stage embedder for German queries.
- Mine hard negatives for embedder training (`reranker.mine_hard_negatives`).
- Provide soft labels / margins for distillation (`distillation_soft_labels`, `margin_mse_target`).

## Usage
```python
# Cross-encoder scoring (requires the trained model + transformers)
from boldt_embed.reranker import Reranker
rr = Reranker.from_config("configs/training_reranker.json")
query = "Wie hoch darf die Mietkaution sein?"
docs = ["Die Mietkaution darf höchstens drei Nettokaltmieten betragen.",
        "Die Maklerprovision beträgt häufig zwei Nettokaltmieten."]
ranked = rr.rerank(query, docs)   # [(index, score), ...] best-first
```

## Training
- Input template (`configs/training_reranker.json`):
  `Anfrage: {query}\nDokument: {document}\nIst das Dokument relevant für die Anfrage?`
- Labels `Ja`/`Nein`; hard negatives from BM25, the embedder, and reranker-mined sources.
- Dry-run: `make dry-run-reranker`.

## Evaluation
**Pending.** Report reranking gains (nDCG@10 before/after rerank) on the German suite with
run metadata (ADR-005). No numbers until a saved run exists under `outputs/`.

## Limitations
- Higher latency than the bi-encoders (full cross-attention per pair) — use as a re-ranking
  stage over a shortlist, not for first-stage retrieval over a large corpus.
- 350M German-first; quality unverified until trained and benchmarked.
- `max_length` (1536) bounds combined query+document length.

## License
- **Code:** Apache-2.0. **Base weights:** apache-2.0 (verified 2026-05-28).
- **Derivative weights:** intended apache-2.0, contingent on training-data licenses (ADR-004).

## Reproducibility
- Template, labels, and mining/distillation interfaces are pinned in `configs/` and `reranker.py`.
- Validate: `make all`. Dry-run: `make dry-run-reranker`. Record commit + run metadata for evals.
