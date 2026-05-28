# Boldt-Embed-DE — Implementation Repo Instructions

This is the **implementation** repository for the German-first embedding model family
based on `Boldt/Boldt-DC-350M`. It was bootstrapped from the Boldt prompt pack.

## Mission

Ship three auditable, reproducible artifacts:

1. `Boldt-Embed-DE-350M-v1-causal`
2. `Boldt-Embed-DE-350M-v1-bi`
3. `Boldt-Reranker-DE-350M-v1`

…together with research notes, ADRs, a data/license/leakage plan, training code,
an evaluation harness, validation gates, model cards, and a final audit.

## Non-negotiable rules

- The importable core and all validation gates MUST run on the Python standard library
  only. `torch`/`transformers` are optional extras, imported lazily.
- Inspect before editing. Keep a task list. Small, reviewable commits per milestone.
- **Never claim a benchmark result unless the command was run and its output saved under
  `outputs/` with run metadata.** Separate toy/local plumbing from real model benchmarks.
- Do not commit model weights, large datasets, or secrets.
- Keep licensing and benchmark-leakage concerns visible (ADR-001, ADR-004, ADR-005).
- German-first design: query/task instructions, Matryoshka dims, and German hard
  negatives (compounds, negation, legal refs, dates/numbers, regional variants, entities).

## Validation (run before claiming a milestone is done)

```bash
python scripts/validate_repo.py --format markdown
python scripts/run_smoke_tests.py --format markdown
python scripts/run_local_benchmark.py --format markdown
python -m unittest discover -s tests
```

## Progress report format

Files changed · Commands run · Validation · Benchmark · Latest commit · Working tree · Risks.
