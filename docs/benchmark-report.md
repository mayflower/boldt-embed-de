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
