# ADR-001 — Base model and license

## Status
Accepted (2026-05-28); arch/size **VERIFIED 2026-05-29** by loading on GPU: LlamaForCausalLM,
hidden 1024, 24 layers, vocab 32000, ctx 2048, **~435M total params** (publish ~435M honestly).

## Context
All three variants build on `Boldt/Boldt-DC-350M`. License and provenance gate any release.
Verified from the HF model card on 2026-05-28:
- Base-weight license: **`apache-2.0`**.
- German base LM trained on the FineWeb-2 German "Dense-Core" subset (~200B tokens), BF16,
  *not* instruction-tuned.
- Architecture internals (layers, hidden size, vocab, context length) are **not stated**.
- The repo name says "350M" while the HF size badge says "0.5B".

## Decision
1. Use `Boldt/Boldt-DC-350M` as the base for the causal, bidirectional, and reranker variants.
2. License **this repository's source code** under Apache-2.0.
3. Treat **model-weight licensing separately** from code: derivative weights inherit the
   base `apache-2.0` terms, but each *added training dataset* carries its own license
   (resolved in ADR-004). Do not state a weights license in a model card until both the
   base terms and every data source are confirmed.
4. Before publishing a model name/size, **verify the true parameter count** and either
   adopt the accurate figure or document the discrepancy explicitly.

## Consequences
- Apache-2.0 base removes the biggest release blocker; remaining risk shifts to *data* licensing.
- We must read `config.json`/`tokenizer_config.json` from the base repo before finalizing
  pooling and output dims (ADR-003) — tracked as a MUST-VERIFY in the research notes.
- Model cards must cite the base model, its license, and the verification date.

## Alternatives
- **Train a German embedder from scratch:** far costlier and weaker than adapting a strong
  German base. Rejected.
- **Use a multilingual base (e5/bge/Qwen):** not German-first; different licensing/quality
  tradeoffs. Kept only as evaluation baselines, not as our base.

## Test/benchmark criteria
- Base config loaded on GPU and arch/size/license verified (done 2026-05-29).
- Release checklist gates the *data* and *derivative-weights* license before any publish.
