# Benchmark Report (prompt 13)

Commit context: see `outputs/real-training/*.json` for per-run metadata (command, commit,
gpu, torch, date). Generated from executed runs; unrun items say **not run** with the blocker.

## 1. Executive summary
The training/eval pipeline is implemented and **verified on real hardware** (NVIDIA RTX
A6000): all three tracks load `Boldt/Boldt-DC-350M` and train for real. **Headline real
result:** fine-tuning the causal embedder on **11,494 real GermanQuAD pairs** (CC-BY-4.0)
lifts held-out test retrieval from near-random to **nDCG@10 = 0.879 / Recall@1 = 0.779**
(2,204 queries vs 474 passages). This is a genuine in-domain German retrieval model. Caveat:
one dataset / small corpus — **not** a broad multi-domain or full-MMTEB claim (still not run).

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

### 6b. Bidirectional (LLM2Vec) — training signal (`bidirectional-report.json`)
| Phase | initial loss | final loss |
|---|---:|---:|
| MNTP (denoising) | 9.31 | 5.45 |
| Contrastive | 3.04 | ~0.00 |
Bidirectional attention verified: token-0 hidden Δ = 0.0 (causal) vs 7.35 (bidirectional).

### 6c. Reranker — reranking eval (`reranker-eval-report.json`)
| Ranking | nDCG@10 | MRR@10 |
|---|---:|---:|
| id-order baseline | 0.478 | 0.322 |
| + reranker (tiny, trained on 7 triples) | 0.477 | 0.322 |
Train pairwise accuracy (pos > hard-neg) = **1.0** (overfits its pairs; no generalization to unseen queries).

### 6d. Eval suite — HashingEncoder STAND-IN (plumbing, `outputs/benchmarks/eval-suite-report.json`)
| Task | Metric | Value |
|---|---|---:|
| STS | Spearman | 0.868 |
| Classification | accuracy | 1.000 |
| Clustering | V-measure | 0.169 |
| Cross-lingual DE→EN | nDCG@10 | 0.598 |
| RAG | nDCG@10 | 0.900 |
| Stress (BM25) | nDCG@10 | 1.000 |

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
