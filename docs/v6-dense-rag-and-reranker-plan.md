# v6 — dense RAG recall + standalone reranker (active product track)

**Status: active.** Supersedes v5. This track resets scope to the **actual product**: a Boldt-based
German **dense RAG embedder** and a Boldt-based German **reranker**, each measured **directly under
the harness** — never via a custom serving policy.

## Why v6 (what v5 proved)

v5 produced useful diagnostics but **not** the product:

- The v5 **RAW reranker** failed its hardness-aware promotion gate (GermanQuAD −0.0285, 16.9%
  catastrophic). Raw always-rerank still fails the guardrails.
- The conservative / abstain / preservation-grid / bounded `margin_override` work is **diagnostic
  only**. No policy-gated serving workaround is shipped, and the frozen bounded policy itself
  **failed** promotion (WebFAQ policy Δ +0.0245 < +0.05; see `benchmark-report.md` §6p).
- **Decisive finding:** the WebFAQ under-lift is **mostly first-stage recall failure** — in 234/344
  failing queries the positive is **absent from the candidate list**, so **no reranker can recover
  it** (`docs/v5-policy-failure-analysis.md`). The bottleneck is recall, not reranking.

Therefore v6 attacks the bottleneck directly: **dense first-stage recall**, plus a **standalone
reranker** evaluated as raw lift over fixed candidate lists.

## Goal

1. **Improve dense first-stage recall.** Train/retune the Boldt dense RAG embedder so its candidate
   lists actually contain the positives (raise candidate-set recall and oracle nDCG@10), so a
   reranker has something to reorder.
2. **Train a standalone reranker.** Measure its quality as **RAW lift over FIXED candidate lists**,
   directly under the harness — no abstention, no bounded policy, no serving wrapper.

The dense embedder and the reranker are promoted **independently**: a strong dense retriever can ship
on its own retrieval gate even while the reranker remains Experimental.

## Non-goals / constraints

- **No legal/admin requirement.** Legal/administrative retrieval is not a product target for this
  track.
- **GerDaLIR is diagnostic-only** — reported for OOD insight, **never** a release blocker or gate.
- **No policy-gated serving** as a product. Policy artifacts stay in the repo strictly as
  diagnostics/analysis; model promotion never depends on them.
- Keep "small + deployable" (256-d Matryoshka retention ≥ 0.95 for the dense embedder).

## Evaluation sets

| set | role | gates? |
|---|---|:--:|
| WebFAQ held-out | primary (recall + reranker lift) | yes |
| local RAG (private) | primary (recall + reranker lift), if present | yes |
| GermanQuAD | guardrail / general QA (do-not-regress) | yes |
| DT-test | guardrail (near-ceiling, do-not-regress) | yes |
| GerDaLIR (legal) | **diagnostic only** | **no** |

## Metrics measured directly under the harness

- **Dense first-stage recall:** candidate-set recall@k and **oracle nDCG@10** of the dense first
  stage (does the positive even appear in the candidate list?), on WebFAQ/local RAG/GermanQuAD/DT-test.
- **Retrieval quality:** nDCG@10 / Recall@k of the dense embedder standalone (not reranked).
- **Standalone reranker:** RAW nDCG@10 lift over the FIXED candidate lists (no policy), with the
  Qwen3-Reranker-8B teacher as the ceiling.
- **Matryoshka retention:** 256-d vs full-dim quality for the dense embedder.

## Promotion gates

**Dense embedder (independent):**
- WebFAQ / local RAG first-stage recall and nDCG@10 improve measurably over the current best
  in-house retriever;
- GermanQuAD / DT-test do not regress beyond the near-ceiling tolerance (−0.005);
- 256-d Matryoshka retention ≥ 0.95.

**Standalone reranker (independent):**
- **RAW** lift over FIXED candidate lists on WebFAQ / local RAG (primary) clears the raw promotion
  gate (`eval/v5_rag_lift_gate.json`-style status `pass`), with GermanQuAD / DT-test do-not-regress
  and catastrophic-drop rate within tolerance;
- enforced by `validate_release_2026.py` (`check_reranker_raw_recommendation`). **Policy-gated
  variants do not count.**

## First steps

1. Quantify the recall gap: candidate-set recall@k + oracle nDCG@10 for the current dense first stage
   on every eval set (this is the v5 §6p bottleneck made measurable).
2. Improve candidate generation / dense retriever so WebFAQ positives stop being absent from lists.
3. Re-train the standalone reranker on the improved (positive-bearing) candidate lists and measure
   RAW lift under the harness.
