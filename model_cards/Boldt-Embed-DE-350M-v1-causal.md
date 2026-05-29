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

### Real GermanQuAD run (held-out test retrieval)
Trained on **11,494 real GermanQuAD pairs** (deepset/germanquad, CC-BY-4.0), 2 epochs / 720
steps, ~6 min on an NVIDIA RTX A6000 (2026-05-29). Evaluated on the **held-out test split**
(2,204 questions vs 474 unique passages), last-token pooling + German query instruction.
Saved: `outputs/real-training/germanquad-report.json` (`scripts/train_causal_germanquad.py`).

| Model | nDCG@10 | MRR@10 | Recall@1 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|---:|
| Base `Boldt-DC-350M` (untrained) | 0.006 | 0.005 | 0.003 | 0.011 | 0.120 |
| **+ contrastive (GermanQuAD)** | **0.879** | **0.851** | **0.779** | **0.963** | **0.995** |

A large improvement, but this is **in-domain** (train and test are both GermanQuAD/Wikipedia)
— it does **not** measure transfer.

### Cross-domain generalization (the honest number) — held-out legal GerDaLIR
Trained on **150k non-benchmark German-Wikipedia pairs** (`deutsche-telekom/wikipedia-22-12-de-dpr`,
CC-BY-SA-4.0), 1 epoch / 4,688 steps, ~49 min on A6000. Evaluated on the **held-out, disjoint-domain
legal** benchmark `mteb/GerDaLIRSmall` (9,969 docs / 12,234 queries); **train↔eval leakage check = 0**.
Saved: `outputs/real-training/disjoint-de-report.json` (`scripts/train_disjoint_de.py`).

| Model | nDCG@10 | MRR@10 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|
| Base `Boldt-DC-350M` (untrained) | 0.0015 | 0.0013 | 0.0028 | 0.018 |
| + contrastive (Wikipedia, 150k) | **0.027** | 0.024 | 0.043 | 0.126 |
| _baseline_ `intfloat/multilingual-e5-base` | _0.153_ | _0.140_ | _0.218_ | _0.404_ |

**Honest interpretation:** ~17× over the useless base, but **absolute quality is low**. The
**baseline** (multilingual-e5-base, same harness, `outputs/real-training/baseline-gerdalir-report.json`)
scores **0.153** — so GerDaLIR is hard *and* our Wikipedia-only model is **~5.6× below a strong
off-the-shelf model**. The in-domain GermanQuAD 0.879 is *not* representative of cross-domain
ability. Closing the gap needs: (1) **domain-diverse training data**, (2) **hard negatives**
(an ANCE-style run is in `scripts/train_hardneg_de.py`), (3) more **scale/epochs**, and (4) the
baseline comparison (now done: e5-base = 0.153).

### Broader public benchmark (MMTEB) — not run
Full MMTEB German + GermanDPR + baseline comparison remains pending. Numbers are reported only
from saved runs (ADR-005). (Earlier toy 7-triple smoke run superseded.)

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
