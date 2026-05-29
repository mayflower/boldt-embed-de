# ADR-002 — Causal vs. bidirectional embedding route

## Status
Accepted (2026-05-28): build **both**, decide the production default on German evaluation.

## Context
A causal decoder can be turned into an embedder cheaply via last-token pooling (E5-Mistral
style), with no attention change. Alternatively, the LLM2Vec recipe (bidirectional attention
+ MNTP + contrastive) generally yields a stronger *encoder* but adds an adaptation phase and
merging complexity. The base model is a causal decoder; either route is feasible.

## Decision
- Implement **both** tracks behind a shared embedder interface and shared losses/eval:
  - Track A (causal): last-token/EOS pooling, contrastive training.
  - Track B (bidirectional): bidirectional attention + MNTP, then contrastive; pooling ablation.
- Choose the **production default** strictly from held-out German benchmark results
  (ADR-005), not from intuition. Until then, the causal track is the lower-risk baseline.

## Consequences
- More code and more training compute (two tracks), but a defensible, evidence-based choice.
- The bidirectional track owns extra moving parts: the MNTP adaptation phase and optional
  checkpoint merging — isolated in `model_bidirectional.py` and its trainer.
- The shared interface lets the evaluation harness score both tracks identically.

## Alternatives
- **Causal-only:** cheapest, but lower encoder ceiling. Kept as the safe baseline.
- **Bidirectional-only:** skips the cheap baseline and bets on the riskier MNTP path. Rejected.

## Test/benchmark criteria
- Decision metric: German MMTEB nDCG@10 / MRR@10 on held-out data. Pick the higher track;
  fall back to causal on a tie (lower risk/latency).
