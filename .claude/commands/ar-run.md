---
description: THE AutoResearch loop — you assess the state, pick the most reasonable next experiment, run it, judge it, repeat
argument-hint: "[rounds] [dry|real]"
allowed-tools: Bash(conda run *) Bash(tail *) Bash(cat *) Read Edit
disable-model-invocation: true
---
# AutoResearch loop

Run the AutoResearch loop yourself, in THIS turn — there is no separate controller, no state file, no
fixed step order. **You are the orchestrator.** Each round you read the measured state off disk,
choose the single most promising next experiment, run it, measure it, and decide whether to keep it.

Parse `$ARGUMENTS`: first token = rounds **N** (default 3); second = `dry` (default — plan/validate,
no GPU) or `real` (run on the A6000). Tell me before a long real sweep (N≥3 real).

**Goal:** improve German retrieval toward **beating the same-size peers** (e5-base 278M, LFM2.5 350M)
on MTEB(deu) retrieval-core (GermanQuAD / GerDaLIR / MIRACL / MLDR), without regressing any task. The
in-loop WebFAQ proxy is a cheap inner signal; the MTEB frontier gate is the authoritative judge.

For each round k = 1..N:

1. **Assess the state from artifacts** (the artifacts ARE the state — nothing to record):
   ```bash
   conda run -n boldtembed python scripts/ar_frontier_status.py --format markdown   # ranked candidates, peer frontier, per-task leaders + gaps
   conda run -n boldtembed python scripts/ar_report.py --format markdown --no-write  # Pareto view + proxy metrics
   ```
   Also note which checkpoints exist (`outputs/v8/*`, `outputs/merged/*`) and what last round changed.

2. **Pick the single most reasonable next move** and state a one-line rationale. The menu (choose by
   what the state says is the gap — don't run them in a fixed order):
   - **tune** — edit `configs/autoresearch/experiments/current.json` (loss/training/data_mixture) and
     run a fast proxy trial (`scripts/ar_loop.py`). Cheapest; good for knob hill-climbing.
   - **data** — build a better/more-balanced leakage-clean mixture from the catalogue
     (`scripts/ar_build_mixture.py`) when composition is the limiter.
   - **specialist** — train one domain expert from the shared warm-start
     (`scripts/ar_train_specialist.py`); needs hard negatives first
     (`scripts/ar_refresh_hardnegatives.py`).
   - **merge** — soup/SLERP/TIES/DARE over complementary specialists
     (`scripts/ar_merge_search.py`) once ≥2 complementary checkpoints exist — the trade-off escape;
     highest value-per-GPU-minute when specialists are on disk.
   - **distill** — listwise-KL from the teacher ranking (`scripts/ar_distill_trial.py`) to sharpen.

3. **Run it.** Dry mode → the tool's `--dry-run` (validates + plans, no GPU). Real mode → add the
   tool's explicit `--real`/`--allow-gpu`/`--allow-checkpoints`/`--allow-merge` flags.

4. **Measure + judge.** Eval any produced checkpoint and gate it:
   ```bash
   conda run -n boldtembed python scripts/ar_mteb_trial.py --model <ckpt> --label <name> [--real --allow-gpu]
   conda run -n boldtembed python scripts/ar_promote.py --candidate <name> --format markdown
   ```
   Keep the move if it raises the frontier aggregate with no per-task regression; otherwise note why
   and try a different lever next round. After any edit to `current.json`, run
   `scripts/check_autoresearch_integrity.py --format json` and revert + stop if it flags anything
   outside `configs/autoresearch/experiments/*.json`.

**Stop early** if a candidate is promotable, OR two rounds bring no frontier improvement, OR integrity
fails. At the end print a compact table (round · move · rationale · result · best-so-far) and the
recommended next move.

**Rules:** leakage is fail-closed (only scanned_clean catalogue sources train); never claim a number
beyond a saved `outputs/mteb/<label>/summary.json` (ADR-005); a real promotion still needs human
review; no weights are committed (everything lands under git-ignored `outputs/`).
