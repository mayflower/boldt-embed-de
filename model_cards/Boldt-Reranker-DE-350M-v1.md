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

> **Status: real-scale, competent in-distribution, not yet robustly general.** Current model
> `reranker-de-v2`: 150k examples (DT-de-dpr positives + **e5-mined hard negatives**), 1 epoch,
> A6000 (`scripts/train_reranker_de.py --neg-source e5`). It is **not** the 7-pair toy and **not**
> the broken v1. It slightly lifts BM25 on held-out in-domain DT-test but still degrades a
> different general dataset (GermanQuAD) — see Evaluation. A robust general reranker needs
> diverse training question-styles/domains + a harder eval.

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
**GENERAL reranking** (`scripts/eval_reranker_general.py`, `reranker-general-report.json`):
rerank BM25 / e5 top-50 first stages, nDCG@10 before → after. The current model (`reranker-de-v2`)
is trained on DT-de-dpr positives + **e5-mined hard negatives** (strong-retriever confusions).

| Eval | BM25 → +reranker | e5 → +reranker |
|---|---|---|
| DT-test (in-domain, held-out) | 0.978 → **0.989** (+0.011) | 0.994 → 0.993 |
| GermanQuAD (different general dataset) | 0.903 → 0.776 | 0.939 → 0.800 |

**Honest status:** competent **in-distribution** (slightly lifts BM25 on held-out DT-test;
neutral vs near-ceiling e5) but **not yet robustly general** — it degrades GermanQuAD (a different
question style). The eval tasks are also near-ceiling (small corpora), so they barely show
reranker value.

**Why two versions:** v1 (hard negatives mined by the *weak* warmup embedder) was catastrophic —
it dragged every first stage to ~0.20 (random), failing even in-domain, because the negatives
were too easy ("relevant vs unrelated"); diagnostic showed it scored relevant 0.999 vs random
0.001 but couldn't separate top-50 confusions. v2 fixes this with **e5-mined hard negatives**.
A robust general reranker still needs **diverse training question-styles/domains** (DT +
GermanQuAD-style + mMARCO/mqa) and a harder eval with room to help.

## Limitations
- Higher latency than the bi-encoders (full cross-attention per pair) — use as a re-ranking
  stage over a shortlist, not for first-stage retrieval over a large corpus.
- 350M German-first; quality unverified until trained and benchmarked.
- `max_length` (1536) bounds combined query+document length.
- **Not legal advice:** relevance scoring over German text (including legal/admin passages)
  is for information retrieval only and is **not legal advice** — verify against primary sources.

## Teacher distillation
Trained in the 2026 teacher→student workflow: `Qwen/Qwen3-Reranker-8B` scores German
(query, document) pairs (`configs/teacher_models.json`) and the student is distilled toward
the teacher (listwise KL over candidate scores + pointwise/pairwise; `docs/reranker-training-2026.md`).
Numbers come only from saved runs with run cards (`docs/experiment-registry.md`).

## Training data provenance
Permissively-licensed, **non-benchmark** German pairs + teacher-filtered hard negatives
(`scripts/build_training_candidates.py`, `scripts/mine_hard_negatives_2026.py`); every
candidate carries `source`, `domain`, `license`. Weights are publishable only if every
dataset's license permits it (ADR-004).

## Leakage policy
GermanQuAD / GerDaLIR / MTEB / MMTEB test data are **evaluation-only** and removed from the
candidate pool by `filter_leakage_against_eval_texts` (ADR-009, `docs/data/leakage-policy.md`).

## German stress tests
Evaluated separately on German-specific hard cases — ß/ss and umlaut variants, compounds,
negation, dates/numbers, legal references (§/Absatz/Satz/SGB/BGB), formal/informal register,
and entity disambiguation (`german_adversarial.py`, `benchmarks/stress_cases_de.jsonl`).

## Reranker lift
Quality is reported **only as lift over a fixed first stage** (nDCG@10 first-stage vs
+reranker on pre-built shortlists), with the Qwen3 reranker teacher as the ceiling
(`scripts/eval_reranker_lift.py`). A reranker that does not lift a fixed candidate set is not
shipped — this is the explicit v1 lesson.

## License
- **Code:** Apache-2.0. **Base weights:** apache-2.0 (verified 2026-05-28).
- **Derivative weights:** intended apache-2.0, contingent on training-data licenses (ADR-004).

## Reproducibility
- Template, labels, and mining/distillation interfaces are pinned in `configs/` and `reranker.py`.
- Validate: `make all`. Dry-run: `make dry-run-reranker`. Record commit + run metadata for evals.
