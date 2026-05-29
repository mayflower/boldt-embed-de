# Architecture Plan

Base for all three variants: `Boldt/Boldt-DC-350M` (German base LM, `apache-2.0`,
verified 2026-05-28 — see `RESEARCH_NOTES_2026.md` and ADR-001).

## Track A — Causal decoder embedder · `Boldt-Embed-DE-350M-v1-causal`
- Attention: **causal** (unchanged from base).
- Pooling: **EOS / last non-pad token** hidden state.
- Output: **1024-d**, **L2-normalized**. Matryoshka dims `[1024,768,512,256,128,64]`.
- Training: contrastive (MultipleNegativesRanking / InfoNCE) with German hard negatives;
  query instruction on queries, light/no template on documents.
- Serving: SentenceTransformers-compatible wrapper + HF model card.

## Track B — Bidirectional adapted embedder · `Boldt-Embed-DE-350M-v1-bi`
- Adaptation: **bidirectional attention** + **MNTP** (LLM2Vec recipe), then contrastive.
- Pooling: ablate **mean / EOS / latent-attention**.
- Output: 1024-d, L2-normalized, same Matryoshka dims.
- Optional: checkpoint **merging** (linear / SLERP / adapter-merge) of MNTP and contrastive stages.

## Track C — Reranker · `Boldt-Reranker-DE-350M-v1`
- Cross-encoder: query + document encoded **together**.
- Output: scalar relevance score (or `Ja`/`Nein` logit).
- Uses: production reranking, hard-negative mining, teacher distillation into the bi-encoders.

## Cross-cutting
- Embeddings are always L2-normalized; Matryoshka prefixes are re-normalized after truncation.
- German hard-negative families: compounds, negation, legal refs, dates/numbers, regional
  variants, entity disambiguation.
- Public benchmarks are held-out evaluation only; train-time validation uses a private dev set.

## ADR index
- [ADR-001](adr/ADR-001-base-model-and-license.md) — base model & license
- [ADR-002](adr/ADR-002-causal-vs-bidirectional.md) — causal vs bidirectional
- [ADR-003](adr/ADR-003-pooling-strategy.md) — pooling strategy
- [ADR-004](adr/ADR-004-training-data-and-licensing.md) — training data & licensing
- [ADR-005](adr/ADR-005-benchmark-protocol.md) — benchmark protocol
- [ADR-006](adr/ADR-006-release-and-model-card.md) — release & model card
- [ADR-007](adr/ADR-007-matryoshka-dimensions.md) — Matryoshka dimensions
- [ADR-008](adr/ADR-008-reranker-architecture.md) — reranker architecture
- [ADR-009](adr/ADR-009-training-evaluation-split.md) — training/evaluation split

## Non-goals for v1
- No long-context (8k/32k) retrieval claim without a trained+evaluated context-extension phase.
- No "best multilingual" claim from a 350M German-native model.
- No private or license-incompatible training data.
- No repeated tuning against public leaderboard test labels.
