# ADR-007 — Matryoshka dimensions

## Status
Accepted (2026-05-28); native dim VERIFIED 2026-05-29 (base hidden_size = 1024).

## Context
We want one model serving multiple vector sizes so downstream stores can trade quality for
cost. Matryoshka Representation Learning (arXiv 2205.13147) trains so that embedding prefixes
remain useful and can be truncated post-hoc.

## Decision
- Native output **1024-d** (= base hidden size; no projection head).
- Matryoshka dims **[1024, 768, 512, 256, 128, 64]**, trained with a Matryoshka loss over the
  contrastive objective.
- Truncated prefixes are **re-normalized (L2)** before similarity (`matryoshka.truncate_normalized`).

## Alternatives
- **Single fixed dim (1024 only):** simpler, but no cost flexibility downstream. Rejected.
- **Separate models per dim:** storage/serving explosion. Rejected.
- **PCA/post-hoc compression:** lossy, not jointly trained; weaker than Matryoshka. Rejected.

## Consequences
- One checkpoint serves all dims; vector-store cost scales with chosen dim.
- Matryoshka does not speed up the model itself — only downstream storage/retrieval.
- Eval must report metrics at every dim (ADR-005).

## Test/benchmark criteria
- Unit: `truncate_normalized` returns unit-norm prefixes (`tests/test_matryoshka.py`).
- Bench: report nDCG@10 at each dim; the "quality cliff" (lowest acceptable dim) is recorded
  in the benchmark report. Local plumbing exercised by `evaluate_hashing(matryoshka_dims=...)`.
