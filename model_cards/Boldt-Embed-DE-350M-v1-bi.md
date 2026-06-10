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

> **Status: real MNTP→bidirectional run executed; competitive with the causal student.** On
> 2026-06-10 the full LLM2Vec recipe ran end-to-end: bidirectional attention **verified** (Δ
> 0.0 causal vs 51.7 bidirectional), **MNTP** pre-adaptation (600 steps), then bidirectional
> contrastive. Result: bi+MNTP reaches **0.848 GermanQuAD / 0.967 DT-test / 0.060 GerDaLIR** —
> it **beats** the causal student in-domain (DT-test 0.967 vs 0.950) and is competitive
> elsewhere. Crucially, **MNTP is essential**: the no-MNTP ablation collapsed to 0.401 on
> DT-test. See Evaluation / `docs/benchmark-report.md` §6g. Production default stays
> evidence-driven (causal has a slight OOD edge at this 300-step budget).

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

### 2026 MNTP→bidirectional ablation — EXECUTED (2026-06-10, RTX A6000)
Two bidirectional students trained identically to the causal one (CachedMNRL + Matryoshka, 300
steps, attention patch re-applied at eval) — without vs with MNTP pre-adaptation (600 steps).
nDCG@10 on held-out sets (`outputs/baselines/real_bi*_*.json`):

| Held-out set | causal | bi (no MNTP) | bi + MNTP | e5-base |
|---|---:|---:|---:|---:|
| GermanQuAD | 0.883 | 0.659 | 0.848 | 0.939 |
| DT-test | 0.950 | 0.401 | 0.967 | 0.994 |
| GerDaLIR (legal) | 0.078 | 0.020 | 0.060 | 0.134 |

**Honest:** **MNTP is essential** — without it, switching a causal model to bidirectional
attention collapses quality (DT-test 0.401); MNTP recovers it (→0.967). **bi+MNTP beats the
causal student in-domain** (DT-test 0.967 vs 0.950), is competitive on GermanQuAD, and slightly
behind on OOD legal. At this 300-step budget neither dominates; production default stays
evidence-driven (causal has a slight OOD edge). See `docs/benchmark-report.md` §6g.

(Numbers reported only from saved runs with metadata + run cards — ADR-005.)

## Limitations
- Adds an MNTP adaptation phase and merging complexity vs. the causal route.
- Bidirectional enablement in this scaffold is best-effort; production should rely on a
  vetted LLM2Vec implementation.
- 350M German-first; no long-context or "best multilingual" claims.
- 1024-d output subject to the same base hidden-size MUST-VERIFY (ADR-003).
- **Not legal advice:** retrieval/similarity over German text (including legal/admin
  passages) is for information access only and is **not legal advice** — verify against
  primary sources.

## Teacher distillation
Trained in the 2026 teacher→student workflow: `Qwen/Qwen3-Embedding-8B` scores German
(query, passage) candidates (`configs/teacher_models.json`) and the bidirectional student
learns to match them (MarginMSE distillation + cached contrastive + Matryoshka, after MNTP
adaptation; `docs/bidirectional-adaptation.md`, `docs/modern-embedding-training.md`). Numbers
come only from saved runs with run cards (`docs/experiment-registry.md`).

## Training data provenance
Permissively-licensed, **non-benchmark** German data — multi-domain candidates (mMARCO-de,
clips/mqa, SWIM-IR, synthetic, German-stress) built and license-tracked by
`scripts/build_training_candidates.py`; every candidate carries `source`, `domain`, `license`.
Weights are publishable only if every dataset's license permits it (ADR-004).

## Leakage policy
GermanQuAD / GerDaLIR / MTEB / MMTEB test data are **evaluation-only** and removed from the
candidate pool by `filter_leakage_against_eval_texts` (ADR-009, `docs/data/leakage-policy.md`).

## German stress tests
Evaluated separately on German-specific hard cases — ß/ss and umlaut variants, compounds,
negation, dates/numbers, legal references (§/Absatz/Satz/SGB/BGB), formal/informal register,
and entity disambiguation (`german_adversarial.py`, `benchmarks/stress_cases_de.jsonl`).

## Matryoshka dimensions
Native 1024-d, truncatable to 768 / 512 / 256 / 128 / 64 (re-normalize after truncation);
per-dimension trade-off reported by the Matryoshka sweep in `scripts/eval_hybrid_retrieval.py`.

## License
- **Code:** Apache-2.0. **Base weights:** apache-2.0 (verified 2026-05-28).
- **Derivative weights:** intended apache-2.0, contingent on training-data licenses (ADR-004).

## Reproducibility
- Recipe, config, and merge methods are pinned in `configs/` and the ADRs.
- Validate: `make all`. Dry-run: `make dry-run-bi`. Record commit + run metadata for evals.
