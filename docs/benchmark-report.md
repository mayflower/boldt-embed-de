# Benchmark Report (prompt 13)

Commit context: see `outputs/real-training/*.json` for per-run metadata (command, commit,
gpu, torch, date). Generated from executed runs; unrun items say **not run** with the blocker.

## 1. Executive summary
The training/eval pipeline is implemented and **verified on real hardware** (NVIDIA RTX
A6000). Two contrasting causal results tell the honest story:
- **In-domain (flattering):** trained on 11,494 GermanQuAD pairs → GermanQuAD-test nDCG@10 =
  **0.879**. But train and test share the dataset/domain → not a generalization measure.
- **Cross-domain (honest):** trained on 150k *non-benchmark* German-Wikipedia pairs (DT-de-dpr)
  → held-out **legal** GerDaLIR (leakage-checked = 0) nDCG@10 = **0.027** (base 0.0015). ~17×
  over base, but **low in absolute terms** — Wikipedia-QA training transfers weakly to legal IR.

**Conclusion:** a working *pipeline* + a narrow Wikipedia-trained causal embedder, **not** a
general German retriever. Closing the gap needs domain-diverse data + hard negatives + scale +
baseline comparisons. Broad MMTEB and baselines remain **not run**.

## 2. Model variants compared
| Variant | Status | Notes |
|---|---|---|
| `…-v1-causal` | trained (tiny) | last-token pooling, EOS-append, InfoNCE + hard negatives |
| `…-v1-bi` | trained (tiny) | LLM2Vec: verified bidirectional attention, MNTP, contrastive |
| `Reranker-…-v1` | trained (tiny) | LlamaForSequenceClassification, BCE on pos/hard-neg |
| Base (untrained) | reference | last-token pooling, no fine-tuning |

## 3. Datasets and splits
- **Train:** `data/samples/toy_triples_de.jsonl` (7 triples). **Eval (toy):**
  `benchmarks/toy_de_retrieval.json` (8 queries / 10 docs) + STS/classification/clustering/
  cross-lingual/RAG/stress sets. **Held-out public:** MMTEB German, GermanDPR/GermanQuAD — *not run*.
- Splits follow ADR-009 (public test never trains).

## 4. Metrics
nDCG@10, MRR@10, Recall@1/10, MAP@10 (retrieval); Spearman (STS); accuracy (classification);
V-measure (clustering); bytes/vector (efficiency).

## 5. Hardware and runtime
NVIDIA RTX A6000 (48GB), CUDA 12.4 driver, torch 2.6.0+cu124. Causal tiny run ≈ seconds;
bidirectional (10 MNTP + 12 contrastive steps) and reranker (15 epochs) ≈ tens of seconds.

## 6. Results tables

### 6a. Causal embedder — REAL GermanQuAD held-out retrieval (`outputs/real-training/germanquad-report.json`)
11,494 train pairs · 2 epochs / 720 steps · A6000 · test: 2,204 queries vs 474 passages.
| Model | nDCG@10 | MRR@10 | Recall@1 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|---:|
| Base `Boldt-DC-350M` (untrained) | 0.006 | 0.005 | 0.003 | 0.011 | 0.120 |
| **+ contrastive (GermanQuAD)** | **0.879** | **0.851** | **0.779** | **0.963** | **0.995** |

(An earlier 7-triple toy smoke run — base 0.774 → 0.94 on an 8-query toy set — is superseded.)

### 6a-bis. Causal embedder — CROSS-DOMAIN held-out (the honest number) (`outputs/real-training/disjoint-de-report.json`)
Train: 150k DT-de-dpr **Wikipedia** pairs · 1 epoch / 4,688 steps · A6000. Eval: held-out
**legal** GerDaLIR (9,969 docs / 12,234 queries) · train↔eval leakage = 0.
| Model | nDCG@10 | MRR@10 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|
| Base `Boldt-DC-350M` (untrained) | 0.0015 | 0.0013 | 0.0028 | 0.018 |
| + contrastive (Wikipedia, 150k) | **0.027** | 0.024 | 0.043 | 0.126 |

This cross-domain number — not the in-domain 0.879 — is the honest indicator: Wikipedia-QA
training generalizes weakly to legal retrieval.

**Baseline (same harness, `outputs/real-training/baseline-gerdalir-report.json`):**
`intfloat/multilingual-e5-base` scores nDCG@10 = **0.153** / Recall@100 = 0.404 on GerDaLIR.
So GerDaLIR is hard (a strong model only reaches 0.15) **and** our 0.027 is ~5.6× below it —
a real gap, not just task difficulty.

### 6a-ter. Hard-negative (ANCE) run — honest negative result (`outputs/real-training/hardneg-de-report.json`)
warmup (in-batch, 1 ep, bs=48) → GPU-mine hard negatives → continue-train (1 ep), evaluated on **both** held-out sets:

| Stage | GerDaLIR (cross-domain legal) | DT-test (same-domain Wikipedia) |
|---|---:|---:|
| base | 0.0015 | 0.0038 |
| warmup (in-batch) | **0.0498** | 0.965 |
| + hard negatives | 0.0459 | **0.970** |

**Findings (straight):** (1) hard negatives did **NOT** help cross-domain — they slightly *hurt*
GerDaLIR (0.0498→0.0459) while marginally helping same-domain DT-test; the Wikipedia-mined
negatives sharpen non-transferable features. (2) **Severe domain overfit:** ~0.97 on held-out
Wikipedia vs ~0.046 on legal — the bottleneck is **domain coverage**, not hard negatives; best
GerDaLIR (0.0498) is still ~3× below e5 (0.153). Improving legal/general retrieval needs
domain-diverse (incl. legal-adjacent) training data, which we exclude to keep GerDaLIR clean.

### 6b. Bidirectional (LLM2Vec) — training signal (`bidirectional-report.json`)
| Phase | initial loss | final loss |
|---|---:|---:|
| MNTP (denoising) | 9.31 | 5.45 |
| Contrastive | 3.04 | ~0.00 |
Bidirectional attention verified: token-0 hidden Δ = 0.0 (causal) vs 7.35 (bidirectional).

### 6c. Reranker — general reranking eval (`reranker-general-report.json`)
Real-scale, two iterations (the original 7-pair toy is superseded):
- **v1** (hard negs from the *weak* warmup embedder, 80k ex): **catastrophic** — dragged every
  first stage to ~0.20 (random), even in-domain. Diagnostic: scored relevant 0.999 vs random
  0.001 but couldn't separate top-50 confusions → negatives too easy.
- **v2** (hard negs from **e5** top-K, 150k ex, 1 ep): fixes it.

v2, nDCG@10 (first stage → +reranker):
| Eval | BM25 | e5 |
|---|---|---|
| DT-test (in-domain, held-out) | 0.978 → **0.989** | 0.994 → 0.993 |
| GermanQuAD (different general dataset) | 0.903 → 0.776 | 0.939 → 0.800 |

**Honest status:** v2 is **competent in-distribution** (lifts BM25 on held-out DT-test; neutral
vs near-ceiling e5) but **not robustly general** — degrades GermanQuAD (different question style).
Eval tasks are near-ceiling (small corpora) so they barely show reranker value. A robust general
reranker needs diverse training question-styles/domains + a harder eval. (Legal GerDaLIR rerank —
`reranker-eval-report.json` — was the wrong yardstick for a general reranker and is dropped.)

### 6d. Eval suite — HashingEncoder STAND-IN (plumbing, `outputs/benchmarks/eval-suite-report.json`)
| Task | Metric | Value |
|---|---|---:|
| STS | Spearman | 0.868 |
| Classification | accuracy | 1.000 |
| Clustering | V-measure | 0.169 |
| Cross-lingual DE→EN | nDCG@10 | 0.598 |
| RAG | nDCG@10 | 0.900 |
| Stress (BM25) | nDCG@10 | 1.000 |

### 6e. REAL 2026 teacher→student run — EXECUTED on RTX A6000 (2026-06-09)

The teacher/student distillation workflow (Prompts 1–12) run end-to-end on real data and GPU.
**Teachers:** `Qwen/Qwen3-Embedding-8B` + `Qwen/Qwen3-Reranker-8B` (both downloaded, bf16).
**Training data:** 3,764 multi-domain, **non-benchmark** German candidates — TED talks
(`ger-backtrans-paraphrase`), Wikipedia (`DT-de-dpr`), synthetic-query wiki (`swim-ir`), and
German-stress adversarial — leakage-filtered against the eval corpora. Both teachers scored
every candidate; the **false-negative filter vetoed 464 of 574** adversarial distractors as
teacher-confirmed near-duplicates (the v1 failure mode, caught). Student = `Boldt/Boldt-DC-350M`
+ mean pooling, trained with **CachedMultipleNegativesRankingLoss + MatryoshkaLoss** (300 steps)
on the teacher-validated positives. Run cards: `outputs/run-cards/real-*.json`; eval reports:
`outputs/baselines/real_{germanquad,dt_test,gerdalir}.json`.

nDCG@10 on **held-out** sets (1,500 queries each; first-stage dense retrieval, cosine):

| Held-out set (domain) | Base `Boldt-DC-350M` (untrained) | **Student `boldt-modern-de`** | `multilingual-e5-base` |
|---|---:|---:|---:|
| GermanQuAD (Wikipedia QA, OOD) | 0.288 | **0.883** | 0.939 |
| DT-test (Wikipedia, in-domain) | 0.223 | **0.950** | 0.994 |
| GerDaLIR (legal, **OOD**) | 0.0021 | **0.0782** | 0.1343 |

Recall@100 (same runs): GermanQuAD 0.529 → **0.997** (e5 0.998); DT-test 0.546 → **0.996**
(e5 1.000); GerDaLIR 0.020 → **0.277** (e5 0.380).

**Honest interpretation.**
- The student is **competitive with multilingual-e5-base** on German Wikipedia-QA and in-domain
  retrieval (0.88 / 0.95 nDCG@10 vs e5's 0.94 / 0.99), from a 350M German-only model — a real,
  large jump over the untrained base (0.29 / 0.22).
- On **out-of-domain legal** (GerDaLIR), the student reaches **0.078** — ~37× the untrained base
  (0.002) and **~1.6–3× better than the v1 Wikipedia-only runs** (0.027 plain / 0.050
  hard-neg; §6 above), now ~58% of e5-base (0.134). Multi-domain teacher-validated training
  measurably improved transfer; e5-base still leads on legal (it has far more, broader training
  data, and we deliberately used **no legal data**, keeping GerDaLIR a clean held-out test).
- This is the genuine result of an **executed** pipeline, not a projection: every number above
  was produced by a saved command with a run card, on the held-out splits, with train↔eval
  leakage filtered.

**What this run did not do** (next steps, not claimed): MNTP-bidirectional student (trained the
causal base with mean pooling instead), MarginMSE score-distillation (the adversarial negatives
were teacher-identified false negatives, so contrastive used in-batch negatives), reranker
training, and a full MMTEB sweep.

### 6f. REAL reranker + Matryoshka + bidirectional (EXECUTED 2026-06-10, RTX A6000)

Continuation of the §6e run on the freed GPU.

**Bidirectional attention — VERIFIED on the real model** (`llm2vec_boldt.verify_bidirectional_attention`,
transformers 5.9): changing the last token of a sequence leaves an early token's hidden state
**unchanged under causal attention (Δ = 0.0)** but **changes it under the bidirectional patch
(Δ = 51.15)** → `is_bidirectional = true`. The LLM2Vec mask patch genuinely works. (A
bidirectional student that stays bidirectional at *eval* time needs the patch persisted on
load — a wrapper change — so the §6e student remains causal+mean-pooling for now.)

**Reranker — trained + lift over a FIXED first stage.** A cross-encoder (`Boldt-DC-350M` +
classification head) trained pointwise (BCE, final loss 0.023) on **3,190 teacher-validated
positives vs 8,692 dense-mined genuine hard negatives** (student-retrieved; the false-negative
filter vetoed 866 where the Qwen3 reranker still rated them relevant — positive reranker score
median **6.94** vs genuine-neg **−5.19**). Evaluated as lift over the student's top-50 dense
first stage (1,000 queries; `scripts/eval_reranker_lift.py`,
`outputs/real-training/reranker-lift-*.json`):

| Held-out set | first-stage nDCG@10 | + student reranker | oracle |
|---|---:|---:|---:|
| DT-test (in-domain) | 0.950 | **0.990** | 0.994 |
| GermanQuAD (different question style) | 0.886 | **0.532** | 0.996 |

**Honest finding:** the reranker **lifts in-distribution** (DT-test +0.040, near oracle) but
**degrades GermanQuAD** (−0.354) — the same v1 lesson, sharper: a cross-encoder trained on one
candidate distribution does not generalize to a different question style. It is real and
correctly trained (clean positive/negative separation), but **competent in-distribution, not
robustly general**. Generalizing needs diverse training question-styles (DT + GermanQuAD-style
+ web) and a harder first stage — not yet done.

**Matryoshka dimension sweep — REAL** (student on held-out GermanQuAD, matmul ranking,
`outputs/baselines/real_matryoshka_germanquad.json`):

| dim | 1024 | 768 | 512 | 256 | 128 | 64 |
|---|---:|---:|---:|---:|---:|---:|
| nDCG@10 | 0.882 | 0.880 | 0.876 | 0.859 | 0.825 | 0.732 |
| Recall@10 | 0.973 | 0.973 | 0.977 | 0.967 | 0.941 | 0.879 |

Quality holds to **256-d (~97% of full at 4× smaller storage)**; the cliff is below 128-d.
This is a real measured trade-off (re-normalize after truncation), not the toy-encoder §7.

### 6g. Bidirectional student + MNTP ablation — EXECUTED (2026-06-10)

Does bidirectional pooling beat the §6e causal mean-pooling student? Two bidirectional
students were trained identically to §6e (CachedMNRL + Matryoshka, 300 steps) with the
LLM2Vec attention patch enabled (eager attention; verified at train **and** eval via
`train_modern.apply_bidirectional_to_st`) — one **without** MNTP, one **with** MNTP
pre-adaptation (`prepare_bidirectional_student.py`, 600 MNTP steps over the candidate
passages, loss 4.53→3.98). nDCG@10 on the held-out sets (`outputs/baselines/real_bi*_*.json`):

| Held-out set | causal (§6e) | bi, **no** MNTP | bi, **+ MNTP** | e5-base |
|---|---:|---:|---:|---:|
| GermanQuAD | 0.883 | 0.659 | **0.848** | 0.939 |
| DT-test | 0.950 | 0.401 | **0.967** | 0.994 |
| GerDaLIR (legal) | 0.078 | 0.020 | **0.060** | 0.134 |

**Findings (clean ablation):**
- **MNTP is essential, not optional.** Flipping a causally-pretrained model to bidirectional
  attention and running only contrastive training **wrecks it** (DT-test 0.950 → 0.401) — its
  representations were learned under causal masking. **MNTP pre-adaptation recovers almost all
  of it** (DT-test 0.401 → **0.967**, GermanQuAD 0.659 → 0.848). This empirically reproduces
  the core LLM2Vec result on Boldt.
- **bi+MNTP vs causal:** bi+MNTP **beats** the causal student in-domain (DT-test 0.967 vs
  0.950), is competitive on GermanQuAD (0.848 vs 0.883), and slightly behind on OOD legal
  (0.060 vs 0.078). At this small (300-step) budget neither dominates; both are real,
  competitive students. The production default stays evidence-driven — currently causal for
  its slight OOD edge, but bi+MNTP is the stronger in-domain retriever.

The bidirectional patch is persisted-on-load (re-applied at eval), so these numbers reflect a
genuinely bidirectional encoder, not a causal one. Run cards: `outputs/run-cards/real-mntp.json`,
`real-train-embedder-bi*.json`, `real-eval-bi*-*.json`.

### 6h. v2 data-scale-generalization — EXECUTED (2026-06-11/12, RTX A6000)

The v2 experiment asks: **does scaling teacher-validated training data and adding domain
diversity improve generalization** (esp. out-of-domain legal) over the §6e/§6g v1 students?

**Data (real, leakage-filtered, PII-scrubbed).** A **44,336-candidate** multi-domain set
(`scripts/acquire_v2_sources.py` → `build_v2_candidates.py`): real HF sources — `dt_de_dpr`
(wiki), `ger_backtrans_paraphrase` (web: TED/news/Europarl), `swim_ir_de` (wiki) — plus the
real adversarial stress set, plus **honestly-synthetic query families generated over REAL
Wikipedia passages** for the domains with no licensed corpus on disk (faq/admin/legal-adjacent/
cross-lingual; documents are real, only query phrasing is templated). Manifest-gated, dedup'd,
PII-dropped (48 rows: ipv4/phone), leakage-filtered vs GermanQuAD+DT-test contexts (20 rows).
Both 8B Qwen3 teachers scored all 44,336 pairs; the reranker-threshold≥2.0 filter kept
**22,181 teacher-validated positives** (~2.4× v1). **Honest finding from the teacher filter:**
the real sources validated at 95–99% but the *synthetic-query* domains mostly did **not**
(admin 4.8%, faq 5.7% ≥2.0) — templated queries over wiki passages are weak matches, so the
effective training set is web+wiki-dominated.

**Causal student (same recipe as §6e, 600 steps on the larger set), nDCG@10:**

| Held-out set | base | v1 causal (§6e) | **v2 causal** | v2 success min | e5-base |
|---|---:|---:|---:|---:|---:|
| GermanQuAD | 0.288 | 0.883 | **0.886** | 0.88 ✓ | 0.939 |
| DT-test | 0.223 | 0.950 | **0.944** | 0.95 ✗ | 0.994 |
| GerDaLIR (legal, OOD) | 0.003 | 0.078 | **0.110** | 0.10 ✓ | 0.134 |

**Headline (honest):** more teacher-validated data + domain diversity left **in-domain flat**
(GermanQuAD +0.003, DT-test −0.006 — within noise) but **improved OOD legal generalization
+41% relative (0.078 → 0.110)**, crossing the 0.10 target and narrowing the e5 gap (58%→72%
of e5). This is the v2 goal realized, with the cost that the synthetic in-domain queries did
not survive teacher validation. Matryoshka 256-d retention **0.972** (≥0.95 ✓).

**Bidirectional (bi+MNTP, 600 MNTP + 600 contrastive steps), nDCG@10:** 0.815 / 0.870 / 0.096
— **underperforms v2 causal on all three.** v2's MNTP texts include noisier multi-domain
documents than v1's clean wiki; causal stays the production default.

**Reranker (mixed loss over distribution-aware candidate lists), lift over a fixed first stage:**

| Held-out set | first stage | + v1 reranker (§6f) | + **v2 reranker** |
|---|---:|---:|---:|
| DT-test (in-domain) | 0.950 | 0.990 (+0.040) | 0.985 (**+0.035**) |
| GermanQuAD (diff. style) | 0.886 | 0.532 (**−0.354**) | 0.847 (**−0.040**) |

**The v2 reranker fix worked but is incomplete:** multi-source candidate lists cut the
cross-distribution GermanQuAD degradation **~9×** (−0.354 → −0.040), yet it is still negative,
so the **promotion gate FAILS** (`check_reranker_promotion_gate.py`) and the reranker stays
*not recommended*. Likely cause: the capped candidate-list pool inherits the weak synthetic
positives (pos teacher-median 5.0 ≤ neg median 5.7), so labels are noisy.

**Verdict (`summarize_v2_results.py`): MIXED — 3/5 criteria, reranker gate fail.** Release gate
`validate_release_2026.py --require-v2-artifacts` is **red** (reranker promotion) — **not
release-ready**, by design. Run cards: `outputs/run-cards/v2-*.json`; artifacts under
`outputs/v2-generalization/` (`V2_RESULTS.md`, `eval/dense_*.json`, `reranker-lift-*-v2.json`,
`real_matryoshka_germanquad.json`).

**Known v2 inefficiencies surfaced (impl-repo follow-ups):** pure-Python `data.find_leakage`
and `negative_mining_2026.bm25_rank` (rebuilds the index per query) are O(n·m) and do not scale
to the full candidate set — leakage was filtered against GermanQuAD+DT only, and hard-neg /
reranker-list mining was run over a domain-balanced ~3.5k subset (logged, not silently capped).

### 6i. v3 real-domain (real FAQ) — EXECUTED (2026-06-14, RTX A6000)

The v3 question: do **real** licensed domain corpora succeed where v2's synthetic queries failed?
Sourced **real German FAQ** — `PaDaS-Lab/webfaq` (deu, CC-BY-4.0 verified) — and built a 24,761
real candidate set (faq 6000 REAL + web 10000 + wiki 8000 + stress 761), 0 leakage hits, then
ran the full pipeline (both 8B teachers → calibrate → causal student, 600 steps).

**Headline finding — real FAQ passes teacher validation:**

| domain | teacher accept ≥2.0 | v2 comparison |
|---|---:|---|
| **faq_real (real WebFAQ)** | **70.8%** | v2 SYNTHETIC faq: **5.7%** |
| web / wiki / stress | 98.8 / 98.2 / 98.6% | — |

The v2 admin/faq/legal failure was the **synthetic queries**, not the domain — real FAQ data
validates ~12× better. Dense nDCG@10 (`outputs/v3-real-domain/eval/dense_*.json`):

| Held-out set | base | v1 | v2 | **v3** | e5-base |
|---|---:|---:|---:|---:|---:|
| GermanQuAD | 0.288 | 0.883 | 0.886 | **0.885** | 0.939 |
| DT-test | 0.223 | 0.950 | 0.944 | **0.970** | 0.994 |
| GerDaLIR (legal OOD) | 0.003 | 0.078 | 0.110 | **0.089** | 0.153 |

**Honest:** v3 gives the **best DT-test yet (0.970)** and flat GermanQuAD, but GerDaLIR (0.089)
is **below v2 (0.110)** — v3 replaced v2's synthetic legal-adjacent data with FAQ, and FAQ does
not transfer to legal. Verdict **`invalid_for_promotion`** (domain-quality gate): admin_real +
legal_adjacency_real are still unsourced, and faq_real (4,248 accepted) is below the 5,000 floor.
Real legal/admin pairs are the remaining blocker. Run cards: `outputs/run-cards/v3-*.json`.

### 6j. Track transition → v4 RAG reranker (2026-06-14)

**v3's legal/admin domain gates are no longer the active product target; v4 optimizes RAG
reranker quality.** v3 stays as **historical/diagnostic** — its dense causal student is the
current best causal retriever (DT-test 0.970), and its finding (real FAQ validates at 70.8% vs
synthetic 5.7%) is the basis for v4. From here, **GerDaLIR (legal) is a DIAGNOSTIC only**, never
a release blocker. The active target is a German **RAG reranker** that lifts fixed first-stage
candidate sets (WebFAQ-held-out + local RAG +0.03, GermanQuAD/DT-test neutral-or-better, no
catastrophic degradation). See `docs/v4-rag-reranker-plan.md` and
`configs/experiments/v4_rag_reranker.json`.

**v4 promotion is mechanical.** `scripts/summarize_v4_rag_results.py` writes
`outputs/v4-rag-reranker/V4_RAG_RESULTS.{md,json}` with the verdict (promoted / mixed /
not_promoted), and `validate_release_2026.py --require-v4-rag-artifacts` requires the v4 config,
WebFAQ eval split, fixed candidate lists, teacher-scored lists, lift reports, and the promotion
gate — and forbids the reranker card from claiming "Recommended for German FAQ/RAG reranking"
unless the gate report says pass. GerDaLIR is ignored by this track.

### 6k. v4 RAG reranker — EXECUTED (2026-06-14, RTX A6000, `outputs/v4-rag-reranker`)

Distilled `boldt-rag-reranker-v4` (350M, causal v3 backbone + fresh score head) from
`Qwen/Qwen3-Reranker-8B`. Training supervision: **7,415** WebFAQ teacher-scored candidate lists
(**147,582** pairs scored: 7,415 gold positives / 125,567 hard negatives / 14,600 uncertain),
train↔eval leakage-disjoint by deterministic hash split. First stage = BM25 top-20; quality is
**lift over the fixed candidate list** (nDCG@10 first-stage → +reranker):

| eval set | first-stage nDCG@10 | + reranker | Δ | first-stage recall@10 / oracle | gate check |
|---|--:|--:|--:|--:|:--|
| **WebFAQ held-out** (in-domain) | 0.5945 | **0.8852** | **+0.2907** | 0.648 / 1.0 | pass (Δ ≥ +0.03, recall ≥ 0.5) |
| GermanQuAD | 0.9058 | 0.8347 | **−0.0711** | 0.961 / 1.0 | **fail** (neutral & catastrophic) |
| DT-test | 0.9774 | 0.9767 | −0.0007 | 0.992 / 1.0 | fail neutral (not catastrophic) |

**Promotion gate: FAIL → not promoted (verdict: mixed).** The reranker is a **strong in-domain
FAQ reranker** (+0.29 nDCG@10 on held-out WebFAQ) that **does not generalize**: GermanQuAD/DT-test
first stages are already near-ceiling (positive_in_top_10 0.96–0.99, oracle 1.0), so a FAQ-tuned
cross-encoder only churns near-perfect orderings — neutral at best (DT-test), harmful at worst
(GermanQuAD −0.07). This repeats the v1/v2/v3 lesson: in-distribution lift ≠ general reranking.
Next data needed to promote: diverse non-FAQ German question styles (QA-passage, long-doc) in the
teacher-scored training mix, not only WebFAQ FAQ pairs. GerDaLIR/legal stays diagnostic-only and
never gates this track.

### 6l. Track transition → v5 small RAG (2026-06-14)

**v4 is CLOSED; the active product target is `v5-small-rag`** (`docs/v5-small-rag-plan.md`,
`configs/experiments/v5_small_rag.json`, validated by `src/boldt_embed/v5_rag_config.py`). v4
delivered a strong in-domain FAQ reranker (+0.2907 nDCG@10 on held-out WebFAQ) that **did not
generalize** to GermanQuAD (−0.0711); the gate correctly blocked promotion. Legal/admin and
**GerDaLIR remain diagnostic-only** — never a release blocker for this track.

v5 encodes the two v4 lessons as hard rules:

1. **Diverse training styles, not single-style FAQ.** Train on FAQ, QA-passage (non-eval), web
   non-FAQ, long-doc chunks, German stress, and local RAG — with **hardness-aware** mixed-first-
   stage candidate lists (in-house dense + e5 + bge-m3 + BM25), so reranker skill is measured on
   non-trivial lists rather than near-perfect BM25 ones.
2. **Near-ceiling sets are not the promotion signal.** GermanQuAD/DT-test first stages are
   near-ceiling (oracle nDCG@10 = 1.0); reranking can only churn them. v5 treats any set with
   oracle ≥ 0.98 as **do-not-regress** (−0.005 tolerance), **never** a primary promotion driver.
   Promotion is driven by sets with real headroom: WebFAQ held-out ≥ +0.05, local RAG and a hard
   private web-QA set ≥ +0.03, plus a **256-dim Matryoshka retention ≥ 0.95** gate so the
   retriever stays small/deployable. Public benchmarks stay eval-only; no public set may train.

### 6m. v5 small-RAG reranker — EXECUTED (2026-06-15, RTX A6000, `outputs/v5-small-rag`)

First real v5 run (prompt-4 reranker). Multi-domain training data, leakage-filtered against the
guardrails, teacher-scored by Qwen3-Reranker-8B, listwise-KL trained on `Boldt/Boldt-DC-350M`.

- **Data (real):** 6,500 rows — faq_real 2,000 (WebFAQ), qa_passage_non_eval 2,500, german_stress
  1,200, long_doc_chunks 800 (last three from `deutsche-telekom/wikipedia-22-12-de-dpr` **train**
  split, real questions). `web_nonfaq`/`local_rag` omitted (no real source — not faked). Leakage-
  filtered vs dt_test + GermanQuAD (DPR train↔test disjoint, 0 dropped). 5,660 candidate lists
  (BM25 recall 0.871), **FAQ share 0.217**.
- **Teacher scoring:** 113,145 pairs (Qwen3-Reranker-8B); 5,660 gold / 104,091 hard-neg / 3,394
  uncertain. Score separation by domain: faq_real 10.0, german_stress 15.4, long_doc 15.4,
  qa_passage 15.9.
- **Training:** listwise-KL primary on 5,658 queries.

Hardness-aware gate (nDCG@10 over FIXED candidate lists):

| eval set | role | overall Δ | medium+hard | no_room | catastrophic | result |
|---|---|--:|--:|--:|--:|:--|
| WebFAQ held-out (1,360, leakage-filtered) | primary | +0.1665 | +0.370 | 0.54 | 0.010 | pass |
| GermanQuAD | guardrail | **−0.0285** | +0.346 | 0.84 | **0.169** | **FAIL** |
| DT-test | guardrail | +0.0211 | +0.542 | 0.96 | 0.000 | pass |

**Verdict: gate FAIL → not promoted.** On the identical fixed guardrail lists, v5 **improves on v4**
(GermanQuAD −0.0711 → −0.0285; DT-test −0.0007 → +0.0211) and lifts every set strongly where there
is real headroom (medium+hard +0.35 to +0.54, including GermanQuAD). But it still **over-reranks
near-ceiling GermanQuAD lists** (84% no_room), netting −0.0285 with 16.9% catastrophic per-query
drops — beyond even the lenient −0.005 near-ceiling tolerance. The reranker stays **Experimental,
not recommended**; next step is **rerank-or-abstain calibration** on confident first stages. See
`outputs/v5-small-rag/V5_RESULTS.md`.

### 6n. v5 conservative reranker (rank-preservation loss) — EXECUTED (2026-06-15, RTX A6000)

Trained a conservative reranker (`boldt-rag-reranker-v5-conservative`) that adds a **rank-
preservation penalty** on high-first-stage-confidence lists (listwise-KL primary +
`lambda_preserve=0.2 * rank_preservation_loss`; 2,265/5,658 = 40% high-confidence; no new teacher
calls). Scored the SAME fixed eval lists (77k pairs) and ran the SAME hardness gate + abstention
eval. GermanQuAD, apples-to-apples:

| approach | GermanQuAD overall | GermanQuAD catastrophic | WebFAQ overall | DT-test overall |
|---|--:|--:|--:|--:|
| v5 raw (always-rerank) | −0.0285 | 0.169 | +0.1665 | +0.0211 |
| abstain only | −0.0016 | 0.103 | +0.1285 | +0.0180 |
| conservative only | **+0.0094** | 0.122 | +0.1379 | +0.0212 |
| conservative + abstain | **+0.0243** | **0.074** | +0.0975 | +0.0193 |

**Real measured progress, NOT promoted.** The rank-preservation objective turns GermanQuAD overall
**positive** (−0.0285 → +0.0094 / +0.0243) and reduces catastrophic drops (0.169 → 0.122 → 0.074),
while WebFAQ and DT-test stay healthy. The gate **still fails**: the remaining failure is a
**catastrophic tail risk on near-ceiling GermanQuAD lists** — conservative-only trips
`germanquad_catastrophic` (0.122 > 0.05) and conservative+abstain trips it (0.074 > 0.03) plus
marginally `dt_test_beats_always_rerank`. The original −0.07 regression is now a small residual
tail. The reranker stays **Experimental / not recommended**. These conservative/abstain variants are
**diagnostics only**, not the product — see the scope reset in §6p. `outputs/v5-small-rag/V5_RESULTS.md`.

### 6o. v5 conservative preservation grid — negative training result, positive policy confirmation (2026-06-15)

Tested the hypothesis "stronger preservation on the highest-risk lists fixes catastrophic at the
MODEL level." Trained 3 variants (preserve_top_k=3, teacher_margin_override=3.0), scored each on the
same fixed eval lists, compared apples-to-apples to the original conservative checkpoint.

| checkpoint | λ / hc-pct | RAW always-rerank GQ catastrophic | RAW GQ Δ | RAW WebFAQ Δ | bounded(margin_override) GQ catastrophic | bounded gate |
|---|---|--:|--:|--:|--:|:--|
| conservative (orig) | 0.2 / 0.60 | 0.123 | +0.009 | +0.140 | 0.015 | pass |
| lp04 | 0.4 / 0.70 | 0.175 | −0.029 | +0.156 | 0.028 | pass |
| lp06 | 0.6 / 0.75 | 0.137 | −0.002 | +0.144 | 0.019 | pass |
| lp08 | 0.8 / 0.80 | 0.112 | +0.018 | +0.196 | 0.015 | pass |

**Negative training result, positive policy confirmation.** No λ makes raw always-rerank safe —
GermanQuAD catastrophic stays 0.11–0.18 (lp04 *worse*, and pushed raw GQ overall negative); none
approach the 0.03 bar without a policy. Root cause: preservation protects WebFAQ-gap-defined
high-confidence lists during training, which do not transfer to GermanQuAD's near-ceiling lists —
the same transfer gap that broke the fitted policy threshold. **WebFAQ lift did NOT collapse** (lp08
even improved it to +0.196), so none are "too conservative". **Bounded `margin_override` passes the
gate on every checkpoint including the original**, so retraining did not change the deployable
answer. **No new checkpoint promoted** (lp04/lp06/lp08); the original conservative checkpoint plus
the bounded experiment was the best *diagnostic* candidate. The bounded policy was then frozen and
validated on a held-out near-ceiling guardrail — see §6p. `outputs/v5-small-rag/grid/grid_comparison.md`.

### 6p. Frozen bounded-policy validation FAILED → scope reset to v6 (dense recall + standalone reranker)

The frozen bounded `margin_override` policy was evaluated against its promotion gate on a held-out,
train-disjoint **near-ceiling guardrail** (716 WebFAQ lists) plus WebFAQ/GermanQuAD/DT-test. Result
(`outputs/v5-small-rag/policy/promotion_gate.md`):

| eval set | role | policy Δ | raw Δ | catastrophic | gate |
|---|---|--:|--:|--:|:--|
| webfaq | primary (lift) | **+0.0245** | +0.1396 | 0.0007 | **fail (< +0.05)** |
| near_ceiling | primary guardrail | −0.0005 | −0.0088 | 0.0014 | pass |
| germanquad | guardrail | +0.0369 | +0.0087 | 0.0033 | pass |
| dt_test | guardrail | +0.0175 | +0.0212 | 0.0000 | pass |

**The bounded policy FAILED its promotion gate** on exactly one condition — the WebFAQ lift bar. The
policy is a safe do-no-harm wrapper (every guardrail passes; it beats raw on GermanQuAD and
near-ceiling), but the bounds that guarantee safety throttle WebFAQ lift below +0.05. Failure
analysis (`outputs/v5-small-rag/policy/failure_analysis.md`, `docs/v5-policy-failure-analysis.md`)
shows the WebFAQ under-lift is **dominated (234/344) by first-stage recall failure**: the positive is
**absent from the candidate list** (BM25 never retrieved it; first-stage nDCG = 0), so **no
reranker — raw or bounded — can recover it.**

**Scope reset.** The policy/bounded-rerank work is **diagnostic only and not the product**; we do not
ship a policy-gated serving workaround. The actual product is a **Boldt dense German RAG embedder +
a standalone reranker**, with quality measured **directly under the harness**. The next active track
is **dense first-stage recall + standalone reranker quality** — see
`docs/v6-dense-rag-and-reranker-plan.md`. v5 is closed: reranker stays Experimental / not recommended.

## 7. Matryoshka truncation analysis
Storage scales linearly with dim (fp32): 1024→4096 B, 512→2048 B, 256→1024 B, 128→512 B,
64→256 B/vector. The HashingEncoder by-dim retrieval (toy) stays near-perfect down to 64 dims
because the toy queries are lexically trivial — a *real* model's quality cliff must be measured
on MMTEB (not run). Truncated vectors are re-normalized before cosine.

## 8. Error analysis (German examples)
- **Reranker generalization:** trained on "Kündigungsfrist/Widerruf/Mietkaution" triples, it
  ranks the 8 *unseen* benchmark queries (e.g. "elster zertifikat abgelaufen") no better than
  id-order — expected from 7 examples. Fix: scale data.
- **Cross-lingual stand-in:** char-n-gram hashing cannot match "Hauptstadt" ↔ "capital", so
  DE→EN nDCG is low (0.60). A real multilingual-capable encoder is required.
- **Clustering stand-in:** char-n-grams cluster by surface form, not topic (V=0.17); the
  "miete/steuer/gesundheit" topics need semantic embeddings.
- **Stress (orthography):** "ss statt ß / Strasse" retrieves correctly because normalization
  folds ß→ss — a deliberate German design choice (`textutil.normalize`).

## 9. Known limitations
Tiny data; no production training; no public MMTEB run; bi/reranker not generalization-tested;
stand-in metrics are plumbing only. See `docs/audit/final-audit.md`.

## 10. Reproducibility appendix
```bash
pip install -e '.[train]'            # torch cu124 build (see README) + transformers
python scripts/run_real_training.py     --device-index 0 --epochs 15
python scripts/run_real_bidirectional.py --device-index 0
python scripts/run_real_reranker.py      --device-index 0
python scripts/run_eval_suite.py --save  # stand-in; add --model <ckpt> for real model
# public MMTEB (NOT RUN here): python scripts/run_mteb_benchmark_template.py --model <export>
```

## v7 EmbedFilter (planned / dry-run only)

Status: **experimental postprocessor, NOT run here.** No numbers are claimed until a real
`outputs/v7-embedfilter/sweep.json` exists, produced by `scripts/eval_embed_filter_sweep.py`
(after building bases with `scripts/build_embed_filter.py`) and judged by
`scripts/check_embedfilter_gate.py`. EmbedFilter competes against prefix Matryoshka at equal
dimensions; see `docs/v7-embedfilter-plan.md`. No production recommendation until the advisory
gate passes.
