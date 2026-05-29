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

> **Status: pipeline proven on GPU; not production-trained.** A *real* training run was
> executed on an RTX A6000 (2026-05-29) — see Evaluation — but only on 7 toy triples. It
> demonstrates the end-to-end GPU pipeline (real forward/pool/contrastive/backward + a real
> checkpoint), **not** a production-quality model. Production training on licensed German
> corpora at scale is still outstanding.

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
**Real tiny run (NOT a public-benchmark claim).** Executed on an NVIDIA RTX A6000 on
2026-05-29 (`scripts/run_real_training.py`; saved to `outputs/real-training/real-training-report.json`).
Trained 15 epochs on 7 toy German triples; evaluated on the 8-query toy retrieval benchmark
with real embeddings (last-token pooling, query instruction):

| Model | nDCG@10 | MRR@10 | Recall@1 |
|---|---:|---:|---:|
| Base `Boldt-DC-350M` (untrained) | 0.774 | 0.698 | 0.50 |
| + contrastive (this tiny run) | 0.938 | 0.917 | 0.875 |

Caveat: 7 training examples → training loss reaches 0 (the 435M model trivially separates
them); the 8-query eval set is tiny. This shows the pipeline trains and improves a *real*
model, **not** production quality.

**Public-benchmark evaluation (MMTEB / GermanDPR) remains pending** — to be run with full run
metadata per ADR-005 once trained on real corpora. Numbers are reported only from saved runs.

## Limitations
- ~435M-param German-first model (LlamaForCausalLM, hidden 1024, 24 layers): not a "best
  multilingual" model.
- **Max context 2048 tokens** (verified) — no long-context (8k/32k) claim.
- Native 1024-d output confirmed (base hidden_size = 1024); no projection head needed.
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
