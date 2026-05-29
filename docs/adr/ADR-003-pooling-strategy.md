# ADR-003 — Pooling strategy and output dimensions

## Status
Accepted (2026-05-28); hidden-size **VERIFIED 2026-05-29**: base hidden_size = 1024, so the
native 1024-d output is correct and **no projection head is required**.

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
- **VERIFIED (2026-05-29):** base hidden size is exactly 1024 (LlamaForCausalLM, 24 layers),
  so the native 1024-d output is used directly — no projection head.

## Alternatives
- **Mean pooling for the causal track:** dilutes the last-token signal the causal LM is
  trained to produce; kept only inside the bidirectional pooling ablation.
- **CLS pooling:** the base has no trained CLS token. Rejected.

## Test/benchmark criteria
- Unit: pooling shape / mask / normalization (`tests/test_pooling.py`).
- Bench: bidirectional pooling ablation chosen by dev-set nDCG@10.
- Matryoshka users must re-normalize truncated vectors; this is stated in the model cards.
