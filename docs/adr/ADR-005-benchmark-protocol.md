# ADR-005 — Benchmark protocol

## Status
Accepted (2026-05-28).

## Context
We must measure German retrieval/STS/clustering quality credibly and avoid overfitting to
public leaderboards. We also need to evaluate Matryoshka prefixes and German stress cases.

## Decision
1. **Primary suite:** MMTEB German tasks + GermanDPR/GermanQuAD retrieval (BEIR format), run
   via the `mteb` package against a SentenceTransformers-compatible export.
2. **Metrics:** nDCG@10, MRR@10, Recall@10/100, MAP@10 (per `configs/evaluation.json`).
3. **Matryoshka:** report metrics at each dim `[1024,768,512,256,128,64]`.
4. **German stress tests:** compounds, negation, legal refs, dates/numbers, regional variants,
   entity disambiguation — reported separately from aggregate scores.
5. **Held-out only:** public benchmarks are post-training evaluation; no tuning against their
   test labels. Train-time validation uses a private dev split.
6. **Provenance:** every reported number is saved under `outputs/` with required metadata —
   command, commit, model, dataset, split, metric, hardware, output path.

## Consequences
- The local toy benchmark validates plumbing only and is explicitly **not** a model-quality claim.
- Real MTEB runs require the trained model + `eval` extras; kept as a runnable scaffold here.
- A number without saved metadata is treated as **not reported**.

## Alternatives
- **Train-time MTEB as the only eval:** overfits the leaderboard. Rejected (held-out only).
- **Single metric (nDCG only):** hides MRR/recall/MAP tradeoffs. Rejected — report the set.

## Test/benchmark criteria
- Every reported number carries metadata (command, commit, model, dataset, split, metric,
  hardware, output_path) or it is treated as not-reported.
- Local plumbing (metrics + Matryoshka) is unit-tested (`tests/test_metrics.py`, `test_eval_harness.py`).
