---
language: de
license: apache-2.0
library_name: sentence-transformers
pipeline_tag: feature-extraction
base_model: Boldt/Boldt-DC-350M
tags: [german, embeddings, retrieval, matryoshka, bidirectional, llm2vec]
---

# Boldt-Embed-DE-350M-v1-bi

German-first **bidirectional** embedder: `Boldt/Boldt-DC-350M` adapted with the LLM2Vec
recipe (bidirectional attention → MNTP → contrastive), with Matryoshka-truncatable vectors.

> **Status: untrained scaffold.** The adaptation/merge plan and pooling ablation are
> implemented and validated; MNTP + contrastive training has not been run here. No quality
> numbers are claimed below.

## Intended use
- Same as the causal variant (German retrieval / similarity / clustering), but using a
  bidirectional encoder that can capture full-sequence context.
- Pooling is selected by ablation (mean / EOS / latent-attention) on a private dev set.
- Embeddings are L2-normalized; Matryoshka prefixes must be re-normalized.

## Usage
```python
from sentence_transformers import SentenceTransformer  # requires the trained model
model = SentenceTransformer("Boldt/Boldt-Embed-DE-350M-v1-bi")
texts = ["Die Kündigungsfrist für eine Mietwohnung beträgt drei Monate."]
emb = model.encode(texts, normalize_embeddings=True)
```

## Training
- Recipe: (1) enable bidirectional attention, (2) MNTP adaptation, (3) contrastive, optionally
  (4) merge checkpoints (linear / SLERP). See `docs/RESEARCH_NOTES_2026.md` and ADR-002.
- Config: `configs/training_bidirectional.json`. Dry-run: `make dry-run-bi`.
- A full implementation should use the `llm2vec` package.

## Evaluation
**Pending** — identical protocol and honesty rule as the causal card (ADR-005). No numbers
are reported until a saved MTEB run with metadata exists under `outputs/`. The causal vs.
bidirectional production default is decided strictly on German benchmark results.

## Limitations
- Adds an MNTP adaptation phase and merging complexity vs. the causal route.
- Bidirectional enablement in this scaffold is best-effort; production should rely on a
  vetted LLM2Vec implementation.
- 350M German-first; no long-context or "best multilingual" claims.
- 1024-d output subject to the same base hidden-size MUST-VERIFY (ADR-003).

## License
- **Code:** Apache-2.0. **Base weights:** apache-2.0 (verified 2026-05-28).
- **Derivative weights:** intended apache-2.0, contingent on training-data licenses (ADR-004).

## Reproducibility
- Recipe, config, and merge methods are pinned in `configs/` and the ADRs.
- Validate: `make all`. Dry-run: `make dry-run-bi`. Record commit + run metadata for evals.
