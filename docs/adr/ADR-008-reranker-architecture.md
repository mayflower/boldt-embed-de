# ADR-008 — Reranker architecture

## Status
Accepted (2026-05-28).

## Context
First-stage bi-encoder retrieval benefits from a second-stage cross-encoder that scores
(query, document) jointly. The reranker also mines hard negatives and acts as a distillation
teacher for the bi-encoders.

## Decision
- **Cross-encoder** on `Boldt/Boldt-DC-350M` via `LlamaForSequenceClassification` (1 logit),
  trained with binary relevance (positive vs hard negative); a `Ja`/`Nein` LM-head variant is
  an option.
- Input template: `Anfrage: {query}\nDokument: {document}\nIst das Dokument relevant für die Anfrage?`
- Used for: production reranking of a shortlist, hard-negative mining, and teacher distillation
  (margin-MSE / KL) into the bi-encoders.

## Alternatives
- **Bi-encoder only (no reranker):** cheaper, lower ceiling on precision. Rejected for quality.
- **monoT5 / seq2seq reranker:** strong but needs an encoder-decoder base; our base is a decoder. Rejected.
- **LambdaMART / lexical reranker:** no semantic understanding of German. Rejected.

## Consequences
- Higher per-pair latency (full cross-attention) → use only over a retrieved shortlist.
- Adds a training pipeline and a distillation interface (`reranker.mine_hard_negatives`,
  `distillation_soft_labels`, `margin_mse_target`).

## Test/benchmark criteria
- Unit: pair formatting, label mapping, score monotonicity, save/load (`tests/test_reranker.py`,
  `tests/test_reranker_real.py`).
- Bench: reranking lift — nDCG@10 of BM25 shortlist before vs after reranking
  (`outputs/real-training/reranker-eval-report.json`).
