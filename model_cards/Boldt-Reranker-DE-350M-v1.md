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

### 2026 reranker (`boldt-reranker-modern`) — EXECUTED (2026-06-10, RTX A6000)
Trained pointwise (BCE, final loss 0.023) on 3,190 teacher-validated positives vs 8,692
**dense-mined genuine hard negatives** (student-retrieved; teacher false-negative filter
vetoed 866 — positive reranker score median 6.94 vs genuine-neg −5.19). Lift over the student's
top-50 dense first stage (1,000 held-out queries), `outputs/real-training/reranker-lift-*.json`:

| Held-out set | first-stage nDCG@10 | + this reranker |
|---|---:|---:|
| DT-test (in-domain) | 0.950 | **0.990** |
| GermanQuAD (different question style) | 0.886 | **0.532** |

**Honest:** lifts in-distribution (DT-test, near oracle 0.994) but **degrades** GermanQuAD —
competent in-distribution, **not robustly general**. Same lesson as v2: generalizing needs
diverse training question-styles + a harder first stage. See `docs/benchmark-report.md` §6f.

## Production default
**Not recommended for general use yet.** The reranker is competent in-distribution but
degrades a different question style (GermanQuAD). It may only be labelled "recommended" once the
**promotion gate** (`scripts/check_reranker_promotion_gate.py`: no GermanQuAD/DT-test
degradation) passes on v2 — enforced by the release gate.

## Known failure modes
- **Domain/question-style shift**: lifts DT-test (0.950→0.990) but degrades GermanQuAD
  (0.886→0.532). Do not deploy as a general reranker until the v2 promotion gate passes.
- **Not legal advice** (see Limitations); use only as a re-ranking stage over a shortlist.

## v4 RAG reranking (German FAQ/RAG)

The current product target is a German **RAG reranker** evaluated as **lift over fixed
first-stage candidate lists** (WebFAQ held-out + local RAG must lift; GermanQuAD/DT-test
neutral-or-better). GerDaLIR (legal) is **diagnostic only** and never gates this track.

**Status: Experimental; not recommended for production reranking.** This status flips to the
recommended-RAG wording **only** when the v4 promotion gate
(`scripts/check_rag_reranker_promotion_gate.py`) passes — mechanically enforced by
`validate_release_2026.py --require-v4-rag-artifacts` (the card may carry the recommended phrase
only if the gate report says pass).

**Measured (2026-06-14, RTX A6000 — `outputs/v4-rag-reranker`, see `V4_RAG_RESULTS.md`):**
distilled from `Qwen/Qwen3-Reranker-8B` over 7,415 WebFAQ teacher-scored candidate lists
(147,582 pairs: 7,415 gold / 125,567 hard-neg / 14,600 uncertain). Lift over **fixed** BM25
top-20 (nDCG@10):

| eval set | first stage | + reranker | Δ | gate check |
|---|--:|--:|--:|:--|
| WebFAQ held-out (in-domain) | 0.5945 | 0.8852 | **+0.2907** | pass |
| GermanQuAD | 0.9058 | 0.8347 | **−0.0711** | **fail** (neutral & catastrophic) |
| DT-test | 0.9774 | 0.9767 | −0.0007 | fail (neutral; not catastrophic) |

**Gate: FAIL → not promoted.** A strong in-domain FAQ reranker that does **not** generalize:
GermanQuAD/DT-test first stages are already near-ceiling (positive_in_top_10 0.96–0.99, oracle
1.0), so the FAQ-tuned reranker only churns near-perfect orderings and loses ground. Keep
disabled for production; usable only for WebFAQ-style FAQ shortlists, as an experiment.

Always true of this reranker:
- it is **not a dense retriever** — it does not search a corpus;
- use it **over candidate lists only** (re-rank a fixed first-stage top-k, not first-stage retrieval);
- it is **evaluated as lift over first-stage candidates** (nDCG@10 delta), not standalone;
- relevance scoring over German text (incl. legal/admin) is **not legal advice**.

## v5 RAG + policy experiments (diagnostic only) — and the v6 scope reset

**Status: Experimental; never recommended — neither raw nor policy-gated.** The v5 RAG reranker
improved on v4 where there is headroom (WebFAQ +0.1665, DT-test +0.0211), but its **RAW always-rerank
promotion gate FAILED** (GermanQuAD −0.0285, 16.9% catastrophic drops). A reranker becomes
recommended only once its **raw** lift over **fixed** candidate lists passes the raw gate — enforced
by `validate_release_2026.py`.

The follow-on serving experiments are **diagnostics only** and are **never** a production
recommendation:

- rerank-or-abstain — diagnostic only; never a production recommendation.
- conservative reranker with a rank-preservation loss — diagnostic only.
- preservation grid (lp04/lp06/lp08) — diagnostic only; no checkpoint promoted.
- the bounded `margin_override` serving experiment — diagnostic only; never a serving recommendation.

We never recommend any policy-gated serving workaround. The frozen bounded experiment was evaluated
on a held-out near-ceiling guardrail and **also failed** its promotion gate (WebFAQ policy
Δ +0.0245 < +0.05). Failure analysis (`docs/v5-policy-failure-analysis.md`) shows the WebFAQ
under-lift is **mostly first-stage recall failure**: in 234/344 failing queries the positive is
**absent from the candidate list**, so **no reranker — raw or bounded — can recover it.**

**Next product target (v6, `docs/v6-dense-rag-and-reranker-plan.md`):** improve **dense first-stage
recall** and train a **standalone reranker** whose quality is measured **directly under the harness**
(raw lift over fixed candidate lists), not via any serving policy. Policy artifacts remain in the repo
strictly as diagnostics/analysis. GerDaLIR (legal) stays diagnostic-only and never gates.

## v6 RAW reranker — EXECUTED (2026-06-16, RTX A6000), gate FAILED → NOT promoted

Recall was fixed first (dense Boldt-v6: WebFAQ Recall@100 0.65 → 0.96), then a standalone reranker
was trained on multi-domain teacher-scored union lists (449,832 Qwen3-Reranker-8B pairs;
positive-absent lists excluded from BCE/pairwise; **no policy loss**) and evaluated **RAW** over fixed
candidate lists (`outputs/v6-reranker/raw_gate.md`):

| eval set | role | raw Δ nDCG@10 | catastrophic | gate |
|---|---|--:|--:|:--|
| webfaq | primary | +0.0358 | 0.050 | fail (< +0.05) |
| germanquad | guardrail | **−0.0864** | **0.207** | fail |
| dt_test | guardrail | +0.0036 | 0.007 | pass |

**Gate FAIL → the reranker is NOT promoted.** It lifts hard/medium queries strongly (GermanQuAD hard
+0.38) but **over-reranks near-ceiling lists** (GermanQuAD no_room −0.127), a model-level failure —
**not** a recall problem (recall is fixed) and **not** something a serving policy may mask. The
reranker becomes recommended **only when the v6 RAW reranker gate passes**
(`scripts/check_v6_raw_reranker_gate.py`); it does not. No serving wrapper is required to make this
model safe — there is no safe-via-wrapper claim, and policy-gated/bounded/abstain results are
diagnostic-only and never promotion evidence.

## License
- **Code:** Apache-2.0. **Base weights:** apache-2.0 (verified 2026-05-28).
- **Derivative weights:** intended apache-2.0, contingent on training-data licenses (ADR-004).

## Reproducibility
- Template, labels, and mining/distillation interfaces are pinned in `configs/` and `reranker.py`.
- Validate: `make all`. Dry-run: `make dry-run-reranker`. Record commit + run metadata for evals.
