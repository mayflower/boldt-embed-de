# ADR-003 — Pooling strategy and output dimensions

## Status
Accepted (2026-05-28). Hidden-size confirmation is a MUST-VERIFY blocker for the 1024-d claim.

## Context
Pooling converts token hidden states into one vector. Causal models favor last-token/EOS
pooling; bidirectional models can use mean or latent-attention pooling. We also commit to
Matryoshka-truncatable embeddings, which requires consistent normalization.

## Decision
- **Causal track:** pool the **last non-pad token** (EOS/last-token).
- **Bidirectional track:** **ablate** mean / EOS / latent-attention pooling; pick by dev-set.
- Always **L2-normalize** the pooled vector. After Matryoshka truncation to a smaller prefix,
  **re-normalize** before computing cosine similarity.
- Target output **1024-d** with Matryoshka dims `[1024,768,512,256,128,64]`.

## Consequences
- Pooling is implemented as pure, mask-aware functions (`pooling.py`) so it is unit-testable
  without weights, and reused by both model wrappers.
- **MUST-VERIFY:** the native 1024-d output assumes the base hidden size ≥ 1024. If the base
  hidden size differs, add a learned projection head to 1024 (and document it). The 1024
  figure is not final until `config.json` is read (ADR-001).
- Matryoshka users must re-normalize truncated vectors; this is stated in the model cards.
