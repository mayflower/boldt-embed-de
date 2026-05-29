# ADR-009 — Training / evaluation split

## Status
Accepted (2026-05-28).

## Context
Embedding models are easy to overfit to public leaderboards. We must separate what trains the
model from what judges it, and guarantee no leakage between them.

## Decision
- **Three disjoint pools:** (a) training pairs/triples, (b) a **private dev split** for
  train-time validation and hyperparameter/pooling selection, (c) **public test** sets
  (MMTEB German, GermanDPR/GermanQuAD) used **only** for final, post-training evaluation.
- Public **test** labels never enter training or dev. Dev is carved from training-domain data,
  not from any public test set.
- Every training batch is leakage-checked against the eval registry (`data.find_leakage`).

## Alternatives
- **Tune on public test (no held-out):** inflates numbers, dishonest. Rejected.
- **Single train/test split, no dev:** no safe model selection signal. Rejected.
- **k-fold CV on tiny data:** high variance for embeddings; deferred until data scales.

## Consequences
- Reported public numbers are defensible (truly held-out).
- Requires maintaining the eval-corpus registry and a dedup step in the data pipeline.

## Test/benchmark criteria
- Unit: leakage detection (exact + near-dup) in `tests/test_data.py`.
- Process: release checklist confirms "no tuning against public test labels" before any claim.
