# ADR-006 — Release and model cards

## Status
Accepted (2026-05-28).

## Context
We publish three artifacts. Releases must be honest, reproducible, and license-clear, and
must not overclaim (no long-context, no "best multilingual" claims from a 350M German model).

## Decision
1. Ship one **model card per variant** (`model_cards/`), each containing: intended use,
   training data summary + licenses, **limitations**, evaluation table **with run metadata**,
   **reproducibility** instructions, and a clear **license** statement (code vs. weights).
2. Gate release on a **checklist** (`RELEASE_CHECKLIST.md`) and the final **audit** (`AUDIT.md`).
3. State every benchmark number with its provenance; omit any number not produced by a saved run.
4. Provide a SentenceTransformers usage example and the exact query/document instruction format.

## Consequences
- Model cards cannot be completed until evaluation produces saved, metadata-tagged results.
- Limitations and non-goals (ADR-005) are surfaced prominently to avoid misuse.
- Release blocked until ADR-001 weight/size questions and ADR-004 data licenses are resolved.
