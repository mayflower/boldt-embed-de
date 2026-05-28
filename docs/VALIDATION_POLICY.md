# Validation Policy

## Gates (run on the Python standard library; no weights, no network)
```bash
python scripts/validate_repo.py --format markdown   # structure + JSON + imports + ADR/card sections
python scripts/run_smoke_tests.py --format markdown # curated deterministic CPU checks
python scripts/run_local_benchmark.py --format markdown --save  # toy retrieval plumbing
python -m unittest discover -s tests                # full unit suite
```
`make all` runs all of the above and writes reports under `outputs/`.

## Every milestone must include
1. Unit tests for new code.
2. A smoke run on tiny local data.
3. A dry run of the relevant training config (`scripts/train_*.py --dry-run`).
4. A benchmark plumbing run.
5. Saved reports under `outputs/`.
6. A git commit with a clean working tree.

## Phase gates
- **Research:** sources listed, claims dated, uncertain facts flagged.
- **Data:** schema validates, licenses tracked, leakage checked, synthetic prompts versioned.
- **Training:** config loads, wrapper instantiates, pooling/loss smoke passes, dry-run passes.
- **Evaluation:** benchmark config loads, toy benchmark passes, MTEB scaffold present, results
  saved with run metadata (command, commit, model, dataset, split, metric, hardware, path).
- **Release:** model cards complete, license + limitations stated, eval table has run metadata,
  reproducibility instructions included.

## Honesty rule
A benchmark number is "reported" only if produced by a saved command under `outputs/` with the
required run metadata. The local benchmark validates plumbing and is never a model-quality claim.
