# v2 — data-scale generalization plan

**v2 is data/candidate-distribution work, not architecture work.** The v1 runs already proved
the techniques (teacher distillation, false-negative filtering, MNTP-bidirectional, Matryoshka)
work on Boldt. What limits quality now is the **size and diversity** of the teacher-validated
training set (3,764 candidates) and the reranker's narrow candidate distribution — both data
problems. v2 scales the candidate pool to 50k–250k across balanced domains and retrains/evals.

Config: `configs/experiments/v2_generalization.json` (loader/validator:
`src/boldt_embed.v2_experiment_config`).

## Current v1 numbers (measured, held-out nDCG@10)

| | causal | bi+MNTP | reranker | e5-base |
|---|---:|---:|---:|---:|
| GermanQuAD | 0.883 | 0.848 | first-stage 0.886 → **0.532** | 0.939 |
| DT-test | 0.950 | **0.967** | first-stage 0.950 → **0.990** | 0.994 |
| GerDaLIR (legal, OOD) | 0.078 | 0.060 | — | 0.134 |

(Matryoshka 256-d ≈ 97% of 1024-d on GermanQuAD.) Causal is the current best embedder; the
reranker degrades GermanQuAD; OOD legal trails e5-base. See `docs/benchmark-report.md` §6e–§6g.

## Domain mix (target fractions, sum = 1.0)

| domain | fraction |
|---|---:|
| web | 0.25 |
| faq | 0.15 |
| admin | 0.15 |
| legal_adjacency_no_eval_overlap | 0.15 |
| wiki_non_eval | 0.15 |
| german_stress | 0.10 |
| cross_lingual_de_en | 0.05 |

## Success criteria (from the config)

- dense GermanQuAD nDCG@10 ≥ 0.88; DT-test ≥ 0.95; GerDaLIR ≥ 0.10 (stretch 0.12).
- reranker GermanQuAD delta ≥ 0.0 (hard floor — no degradation), target +0.02.
- Matryoshka 256-d retention ≥ 0.95.

Two student variants are compared: `causal_mean_pooling` and `bidirectional_mntp`. The reranker
is trained on diverse candidate lists (BM25 + student/e5/teacher dense + teacher reranker) to
fix the v1 generalization failure.

## Non-goals

- **No training on public eval splits.** `public_benchmarks_eval_only` is a hard config error if
  false; GermanQuAD / GerDaLIR / MMTEB stay eval-only and are leakage-filtered out of training.
- **No leaderboard tuning** against public test labels (use a private dev split for iteration).
- **No release claim yet** — v2 is an experiment; release remains gated by `RELEASE_CHECKLIST.md`.

## How it runs (later prompts)

Source manifest (P2) → v2 candidate builder (P3) + synthetic expansion (P4) → sharded teacher
cache (P5) → hard negatives + reranker candidate lists (P6) → train causal & bi+MNTP (P8) →
reranker v2 + anti-degradation gate (P9) → orchestrator (P7) → results dashboard (P10) →
broader eval (P11) → v2 release gate (P12).
