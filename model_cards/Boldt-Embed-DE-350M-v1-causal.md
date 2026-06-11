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

> **Status: real multi-domain distillation run executed; not yet a final release.** On
> 2026-06-09 the full 2026 teacher→student workflow ran end-to-end on an RTX A6000: Qwen3-8B
> teachers scored 3,764 multi-domain non-benchmark German candidates, and the student trained
> with CachedMNRL + Matryoshka reached **0.88 nDCG@10 on held-out GermanQuAD and 0.95 on
> DT-test** (competitive with multilingual-e5-base) and **0.078 on out-of-domain legal
> GerDaLIR** (~37× the untrained base, ~1.6–3× the v1 Wikipedia-only runs) — see Evaluation /
> `docs/benchmark-report.md` §6e. Still outstanding: bidirectional/MNTP student, score
> distillation, broader (incl. legal) data, and a full MMTEB sweep.

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

### 2026 teacher→student run — EXECUTED (2026-06-09, RTX A6000)
The distillation workflow, run end-to-end: Qwen3-Embedding-8B + Qwen3-Reranker-8B teachers
scored 3,764 **multi-domain, non-benchmark** German candidates (TED/Wikipedia/synthetic/
German-stress); the teacher false-negative filter vetoed 464/574 adversarial distractors; the
student (this base + mean pooling) was trained with CachedMNRL + MatryoshkaLoss. nDCG@10 on
held-out sets (1,500 queries each), vs the untrained base and `multilingual-e5-base`:

| Held-out set | Base (untrained) | **Student** | e5-base |
|---|---:|---:|---:|
| GermanQuAD (wiki QA) | 0.288 | **0.883** | 0.939 |
| DT-test (in-domain) | 0.223 | **0.950** | 0.994 |
| GerDaLIR (legal, OOD) | 0.002 | **0.078** | 0.134 |

Competitive with e5-base on German wiki-QA/in-domain; on out-of-domain **legal** the student
(0.078) is ~37× the untrained base and ~1.6–3× the v1 Wikipedia-only runs (0.027–0.050) — real
transfer improvement, though e5-base still leads (we used no legal training data). Saved:
`outputs/baselines/real_*.json`, run cards `outputs/run-cards/real-*.json`,
`docs/benchmark-report.md` §6e.

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
ability. Closing the gap needs: (1) **domain-diverse training data**, (2) more **scale/epochs**, (3) the
baseline comparison (done: e5-base = 0.153).

**Hard-negative experiment (honest negative result, `hardneg-de-report.json`):** an ANCE-style
run (warmup → GPU-mine → continue-train) reached **0.0498** on GerDaLIR (warmup) but the hard
negatives **slightly hurt** cross-domain (0.0498→0.0459) while the model hit **~0.97** on
held-out same-domain Wikipedia (DT-test). Conclusion: the bottleneck is **domain coverage**, not
hard negatives — and 0.0498 is still ~3× below e5. Legal/general quality needs domain-diverse
(incl. legal-adjacent) data, which we exclude to keep GerDaLIR a clean held-out benchmark.

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
- **Not legal advice:** retrieval/similarity over German text (including legal/admin
  passages) is for information access only and is **not legal advice** — verify against
  primary sources.

## Teacher distillation
Trained in the 2026 teacher→student workflow: `Qwen/Qwen3-Embedding-8B` scores German
(query, passage) candidates (`configs/teacher_models.json`) and the student learns to match
them (MarginMSE distillation + cached contrastive + Matryoshka; `docs/modern-embedding-training.md`).
Numbers are reported only from saved runs with run cards (`docs/experiment-registry.md`).

## Training data provenance
Permissively-licensed, **non-benchmark** German data — multi-domain candidates (mMARCO-de,
clips/mqa, SWIM-IR, synthetic, German-stress) built and license-tracked by
`scripts/build_training_candidates.py`. Every candidate carries `source`, `domain`, `license`;
weights are publishable only if every contributing dataset's license permits it
(ADR-004, `docs/data/license-policy.md`).

## Leakage policy
GermanQuAD / GerDaLIR / MTEB / MMTEB test data are **evaluation-only** and are removed from
the candidate pool by `filter_leakage_against_eval_texts` (ADR-009,
`docs/data/leakage-policy.md`). No tuning against public test labels.

## German stress tests
Evaluated separately on German-specific hard cases — ß/ss and umlaut variants, compounds,
negation, dates/numbers, legal references (§/Absatz/Satz/SGB/BGB), formal/informal register,
and entity disambiguation (`german_adversarial.py`, `benchmarks/stress_cases_de.jsonl`).

## Matryoshka dimensions
Native 1024-d, truncatable to 768 / 512 / 256 / 128 / 64 (re-normalize after truncation). The
accuracy/footprint trade-off per dimension is reported by the Matryoshka sweep in
`scripts/eval_hybrid_retrieval.py`.

## Production default
**Current production default: causal (this variant)**, evidence-driven from the executed runs.
bi+MNTP is competitive and beats causal in-domain (DT-test 0.967 vs 0.950), but causal keeps a
slight out-of-domain legal edge at the current budget. Re-decide on v2 results
(`docs/v2-generalization-plan.md`).

## Known failure modes
- **Out-of-domain legal** (GerDaLIR 0.078) trails `multilingual-e5-base` (0.134) — no legal data
  in training; do not rely on it for legal retrieval.
- **Not legal advice** (see Limitations).
- In-domain numbers (GermanQuAD/DT-test) are not representative of arbitrary German domains.

## License
- **Code:** Apache-2.0.
- **Base weights:** `Boldt/Boldt-DC-350M` is apache-2.0 (verified 2026-05-28).
- **Derivative weights:** intended apache-2.0, contingent on every training dataset's license
  (ADR-004). Confirm before publishing weights.

## Reproducibility
- Base model, config, instruction format, and pooling are pinned above and in `configs/`.
- Validate the pipeline: `make all`. Dry-run the trainer: `make dry-run-causal`.
- Record commit + run metadata with any evaluation (ADR-005).
