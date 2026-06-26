# AutoResearch Runbook — the one loop

There is **one** AutoResearch loop and **you (Claude Code) are the orchestrator**. No controller, no
state file, no fixed step order: each round the loop reads the measured state off disk, picks the
single most reasonable next experiment, runs it, measures it, and decides whether to keep it.

## Run it

```
/ar-run            # 3 rounds, dry (plan/validate, no GPU) — see what it would do
/ar-run 5 dry      # more dry rounds
/ar-run 3 real     # 3 real rounds on the A6000 (tell me first for long sweeps)
```

Everything defaults to **dry-run**; real GPU/teacher work happens only in `real` mode (each
underlying tool also has its own `--real`/`--allow-*` flag). The loop reports a running table
(round · move · rationale · result · best-so-far) and stops early on a promotable candidate, two
flat rounds, or an integrity failure.

## What it does each round

1. **Assess state from artifacts** (the artifacts ARE the state):
   `scripts/ar_frontier_status.py` (ranked candidates, peer frontier, per-task leaders/gaps) and
   `scripts/ar_report.py` (Pareto + proxy view); plus which checkpoints exist under `outputs/`.
2. **Pick the single best next lever** (rationale stated), from:
   tune (`ar_loop.py`) · data mix (`ar_build_mixture.py`) · specialist (`ar_train_specialist.py`,
   needs hard negs via `ar_refresh_hardnegatives.py`) · merge (`ar_merge_search.py`) ·
   distill (`ar_distill_trial.py`).
3. **Run it** (dry or real).
4. **Measure + gate:** `ar_mteb_trial.py` then `ar_promote.py` (the protected frontier gate). Keep
   the move if it raises the frontier aggregate with no per-task regression; else try another lever.

## Goal & promotion

Beat the same-size peers (e5-base, LFM2.5) on MTEB(deu) retrieval-core (GermanQuAD / GerDaLIR /
MIRACL / MLDR). A candidate is promotable only if `check_mteb_frontier_gate.py` passes: its 4-task
aggregate ≥ the same-size-peer aggregate, **no** per-task regression below the `v6-1-baseline` floor,
the baseline covers the candidate's tasks, and leakage is clean. Promotion still needs human review;
never claim a number beyond a saved `outputs/mteb/<label>/summary.json` (ADR-005).

## Invariants (unchanged, fail-closed)
- Only `training_usable` + `scanned_clean`/`clean` catalogue sources (`configs/data_sources.json`)
  may train; the single mixture builder enforces it.
- No weights/large corpora committed (git-ignored under `outputs/`).
- Specialists for a merge share the warm-start basin; eval candidates at the same seq length.

## Make targets
```
make autoresearch-report     # Pareto/frontier report across saved artifacts (stdlib, read-only)
make autoresearch-validate   # the AutoResearch tool + recipe unit tests (stdlib, no GPU)
```
