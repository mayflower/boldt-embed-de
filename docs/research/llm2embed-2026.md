# LLM-to-Embedding for Boldt — Research Brief (2026)

- **Compiled:** 2026-05-28 (base-model facts verified on GPU 2026-05-29).
- **Confidence:** `[VERIFIED]` primary source this project · `[LIT]` established literature ·
  `[ASSUMPTION]` not yet confirmed.

This is the canonical research brief required by prompt 02. The dated source notes also live
in `../RESEARCH_NOTES_2026.md`; this file adds the **design-implications table** and **risk list**.

## 1. Verified base model
`Boldt/Boldt-DC-350M` `[VERIFIED 2026-05-29]`: **LlamaForCausalLM**, hidden 1024, 24 layers,
vocab 32000, max_position 2048, **~435M params**, `apache-2.0`, German FineWeb-2 "Dense-Core",
base (not instruct). `AutoModel` exposes `last_hidden_state` for pooling.

## 2. Techniques surveyed
- **Decoder-only causal embedders:** last-token/EOS pooling on the unmodified causal model
  (E5-Mistral style). `[LIT]` arXiv 2401.00368.
- **LLM2Vec / bidirectional adaptation:** (1) replace causal mask with all-ones, (2) MNTP,
  (3) contrastive. `[VERIFIED via source 2026-05-28]` arXiv 2404.05961.
- **Masked-next-token adaptation (MNTP):** blends masked-LM + next-token so the model uses
  both-side context after attention is made bidirectional. `[LIT]`
- **Matryoshka embeddings:** train so prefixes (1024→64) stay useful; reduces vector-store
  cost, not model size; truncated prefixes must be re-normalized. `[LIT]` arXiv 2205.13147.
- **Instruction-aware retrieval:** query instruction on queries, light/no template on docs. `[LIT]`
- **Hard negatives + reranker distillation + merging:** mine negatives (BM25/embedder/reranker);
  cross-encoder teacher distills into the bi-encoder (margin-MSE/KL); merge MNTP+contrastive
  checkpoints (linear/SLERP). `[LIT]`
- **German benchmark/data:** MMTEB (arXiv 2502.13595) + GermanDPR/GermanQuAD (arXiv 2104.12741);
  `mayflower/mteb-de`. Held-out only. `[VERIFIED 2026-05-28]`

## 3. Design implications for Boldt (table)

| Area | Finding | Implication for Boldt-Embed-DE | ADR |
|---|---|---|---|
| Pooling (causal) | last-token works for causal LMs | EOS/last-token pooling, append EOS | ADR-003 |
| Attention (bi) | bidirectional + MNTP gives stronger encoder | build Track B; convert mask + MNTP | ADR-002, ADR-009 |
| Output dim | base hidden = 1024 (verified) | native 1024-d, **no projection head** | ADR-003, ADR-007 |
| Matryoshka | prefix-truncatable, re-normalize | dims 1024/768/512/256/128/64 | ADR-007 |
| Instructions | asymmetric query/doc helps retrieval | German query instruction by default | ADR-003 |
| Hard negatives | dominate contrastive quality | 7 German neg families + reranker mining | ADR-008 |
| Reranker | cross-encoder teacher | train reranker; distill + rerank | ADR-008 |
| Context | base max_position = 2048 | **no long-context claim** | ADR-005 |
| Eval | leaderboard overfit risk | MMTEB held-out; private dev split | ADR-005, ADR-009 |
| License | base apache-2.0; data licenses separate | track per-dataset license; PII filter | ADR-001, ADR-004 |

## 4. Risk list

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Training data license incompatibility | High | allowlist + per-source license registry (ADR-004) |
| R2 | Benchmark leakage into training | High | `find_leakage` vs eval registry; held-out only |
| R3 | PII in scraped German text | Med | PII detector + filter (data.py); document in license-policy |
| R4 | Overclaiming (long-context / multilingual / quality) | Med | non-goals in ADR-005; honest model cards; audit |
| R5 | Bidirectional enablement done wrong | Med | prefer `llm2vec`; attention-mask unit test |
| R6 | Tiny-data overfit mistaken for quality | Med | label real runs "tiny"; gate claims on MMTEB |
| R7 | German morphology (compounds, §, umlauts) hurts tokenization | Med | profile tokenizer fertility; stress tests |

## 5. Open questions (resolve via ablation)
1. Causal vs bidirectional winner on German MMTEB → decides production default (ADR-002).
2. Best bi pooling: mean vs EOS vs latent-attention (ADR-003).
3. Matryoshka quality cliff: lowest dim with acceptable nDCG (ADR-007).
4. Reranker distillation gain vs bi-encoder-only (ADR-008).
5. Tokenizer fertility on German compounds / legal refs (R7).

## Sources
See `../RESEARCH_NOTES_2026.md` §Sources (Boldt card, LLM2Vec 2404.05961, E5 2401.00368,
Matryoshka 2205.13147, MMTEB 2502.13595, GermanQuAD/DPR 2104.12741, mayflower/mteb-de).
