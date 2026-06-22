# v6.1 dense evaluation results

Decides whether the **Boldt dense RAG embedder can be recommended for German RAG first-stage
retrieval**. Based on **dense retrieval quality only** — independent of the reranker (which stays
experimental unless its own raw gate passes) and never gated on policy/bounded/abstain behavior.

- Eval: `scripts/eval_v6_1_dense_top50.py` (`eval_rankings` core) — Gate:
  `scripts/check_v6_1_dense_gate.py` (`dense_gate`). Tests: `tests/test_v6_1_dense_gate.py`.
- Models: dense-v6.1 (candidate), dense-v6, BM25, e5-base (Qwen3-Embedding-0.6B optional, if cached).
- Eval sets: WebFAQ heldout (primary), GermanQuAD + DT-test (guardrails), local RAG (if present);
  **GerDaLIR diagnostic-only, never gates**.

## Dense gate (pass only if all)

| target | threshold |
|---|---|
| WebFAQ Recall@50 | ≥ 0.90 |
| WebFAQ Recall@100 | ≥ 0.96 |
| WebFAQ missing-positive rate | ≤ 0.04 |
| WebFAQ nDCG@10 | ≥ 0.67 |
| GermanQuAD nDCG@10 | ≥ 0.88 |
| DT-test nDCG@10 | ≥ 0.94 |
| Matryoshka-256 retention | ≥ 0.95 |
| public-eval leakage | none |

**If pass:** the dense embedder can be recommended for German RAG first-stage retrieval; the reranker
remains experimental unless the raw reranker gate passes (independent decision).
**If fail:** do not recommend; the gate reports which dense target(s) failed.

## Results — real run 2026-06-16 (`outputs/v6-1-dense-top50/dense_eval_summary.json`)

### WebFAQ (primary)

| model | R@10 | R@50 | R@100 | nDCG@10 | MRR@10 | missing | 256-ret | docs/s |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| BM25 | 0.638 | 0.647 | 0.651 | 0.586 | 0.569 | 0.343 | n/a | n/a |
| e5-base | 0.730 | 0.843 | 0.933 | 0.678 | 0.661 | 0.011 | n/a | 661 |
| dense-v6 | 0.739 | 0.883 | 0.964 | 0.671 | 0.649 | 0.004 | 0.999 | 473 |
| **dense-v6.1** | **0.801** | **0.933** | **0.977** | **0.704** | **0.674** | 0.011 | 1.001 | 412 |

**v6.1 achieved its objective:** Recall@50 0.883 → **0.933** (+0.050, clears 0.90), Recall@100 0.964 →
0.977, nDCG@10 0.671 → 0.704 — best of all four models on WebFAQ. (Note: missing-positive@200 ticked
up 0.004 → 0.011 — a few rank-~150-200 positives slipped past 200 even as the mid-ranks improved;
still well under the 0.04 cap.)

### Guardrails (nDCG@10) — the regression that fails the gate

| model | GermanQuAD | DT-test |
|---|--:|--:|
| dense-v6 | 0.886 | 0.977 |
| **dense-v6.1** | **0.878** | 0.975 |

The WebFAQ-focused rank-promotion cost **−0.008 GermanQuAD nDCG@10** (0.886 → 0.878), landing **0.002
below the 0.88 guardrail floor**. DT-test dipped −0.002 but still passes (0.975 ≥ 0.94).

### Matryoshka (dense-v6.1, nDCG@10 by dim)

| 1024 | 512 | 256 | 128 | 256-retention |
|--:|--:|--:|--:|--:|
| 0.7044 | 0.704 | 0.7049 | 0.696 | **1.001** |

256-d is essentially lossless (retention 1.001).

## Verdict

`outputs/v6-1-dense-top50/dense_gate.{json,md}` — **status: FAIL**. **Do NOT recommend** the dense
embedder yet. Failed target: **`germanquad_ndcg_at_10` (0.878 < 0.88)** — the only failing check;
the WebFAQ objective and all other guardrails/Matryoshka pass. v6.1 traded a small GermanQuAD
regression for a large WebFAQ Recall@50 gain; it clears the floor on everything except GermanQuAD, by
0.002.

> The dense recommendation is **independent of the reranker**: this gate is decided purely on dense
> retrieval quality. v6.1 is **not recommended** because of the GermanQuAD guardrail, not anything
> reranker-related.
