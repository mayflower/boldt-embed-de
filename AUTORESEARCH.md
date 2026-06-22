# AUTORESEARCH — Boldt-Embed-DE dense-retriever experiment loop

Operating manual for future Claude Code / agentic runs. Read this **before** changing anything.

## Goal

Improve **German dense first-stage retrieval** for FAQ/RAG (the v6 product target). This loop
optimizes the **dense retriever only**. Reranker automation is intentionally **deferred** until
dense recall is sufficient — see `09-reranker-later.md` in `prompts.md` and
`AUTORESEARCH_RERANKER.md` (added later).

## Default run budget: **20 minutes**

Every trial defaults to a **20-minute** budget. `scripts/ar_run_trial.py` enforces this:

```bash
conda run -n boldtembed python scripts/ar_run_trial.py \
  --config configs/autoresearch/experiments/current.json \
  --budget-minutes 20 \
  --out outputs/autoresearch/runs/<run_id>
```

- Omitting `--budget-minutes` → 20.
- `--budget-minutes > 20` **fails** unless `--allow-longer-than-20` is passed.
- A longer run, even when allowed, is stamped `invalid_for_default_loop: true` and must not be
  used as default-loop promotion evidence.

Real GPU runs use the **`boldtembed` conda env** on the **RTX A6000** (the 48 GB "RTX 6000",
exposed as the only CUDA device inside that env — `CUDA_VISIBLE_DEVICES=0`). Pass `--real
--allow-gpu` to `ar_run_trial.py` for a real trial; dry-run needs no GPU.

## What you MAY edit (the experiment surface)

- `configs/autoresearch/experiments/*.json` — search space / hyper-parameters per trial.
- `src/boldt_embed/autoresearch_recipe.py` — the dense training/eval recipe.

Nothing else. The recipe is the only Python file the loop normally touches.

## Protected surfaces (never edited by the loop)

`scripts/check_autoresearch_integrity.py` fails the run if a changed path touches any of these:

- evaluation datasets (`data/processed/eval/**`, `outputs/v4-rag-reranker/eval/**`, eval manifests)
- leakage checks (`src/boldt_embed/leakage_index.py`, `scripts/*leakage*`)
- benchmark harnesses (`src/boldt_embed/eval_harness.py`, `eval_v6_1_dense_top50.py`, metrics core)
- scoring scripts (`scripts/ar_score.py`, the gate scripts)
- release gates (`scripts/validate_release_2026.py`, `scripts/check_*gate*.py`)
- baseline outputs (`outputs/autoresearch/baseline/**`, committed baseline reports)
- the **base config** `configs/autoresearch/base_*.json` (it drives every trial via `extends`;
  only the per-experiment overlay `configs/autoresearch/experiments/*.json` is loop-editable)

Run the check before committing. Pass the loop's **start commit** as `--base-ref` so a protected
edit that was *committed* (not just left in the working tree) is still caught:

```bash
conda run -n boldtembed python scripts/check_autoresearch_integrity.py --base-ref <loop-start-sha>
```

## Hard rules (fail-closed)

- **train ≠ eval.** Benchmark sets (GermanQuAD / DT-test / GerDaLIR / MMTEB) are held out.
- **leakage hits must be VERIFIED 0.** Leakage is a property of the data *preparation*: the recipe
  reads the status from the prepared manifest (`--prepared-manifest`) and never fabricates it. The
  scorer **fails closed** — a missing leakage block or a `not_checked`/`unparseable` status fails
  the gate (it is not treated as zero). A trial is promotable only with a verified-clean status.
- **GermanQuAD and DT-test are guardrails**, never the primary signal. They carry only a
  do-not-regress tolerance (`nDCG@10` Δ ≥ −0.005); reranking/over-fitting near-ceiling lists
  only churns them. The primary signal is **WebFAQ held-out** (and local RAG when present).
- **256-d Matryoshka retention** must stay ≥ 0.95.
- **No model weights, checkpoints, HF caches, large datasets, or secrets in git.** Run dirs and
  checkpoints under `outputs/autoresearch/` are git-ignored.
- **No benchmark claim without saved run metadata.** Dry-run metrics are plumbing only and carry
  a `scale_disclaimer`; they are never a quality claim.

## Driving the loop interactively in Claude Code (slash commands)

The loop is instrumented as **project slash commands** under `.claude/commands/`, for use in an
**interactive** Claude Code session. Type them at the prompt:

| Command | What it does |
|---|---|
| `/ar-orient` | Brief on the rules, editable vs protected surfaces, and current integrity status |
| `/ar-status` | Summarize `results.tsv` + the current config; best WebFAQ recall so far |
| `/ar-prepare <args>` | Build the preparation manifest from local data (+ leakage report) |
| `/ar-trial [dry\|real]` | Run ONE iteration (trial → score → log → integrity); report the verdict |
| `/ar-tune [hypothesis]` | Make ONE editable-surface config change toward WebFAQ recall, then iterate |
| `/ar-run [rounds] [dry\|real]` | **Autonomously** run several rounds back-to-back in one invocation |
| `/ar-integrity [--base-ref REF]` | Run the protected-surface check and explain any violations |

A typical interactive session: `/ar-orient` → `/ar-status` → `/ar-trial dry` (sanity) →
`/ar-tune "lower loss.temperature"` (iterate) → `/ar-trial real` once you have a baseline + a
verified-clean prepared manifest. To go hands-free for several rounds at once, use `/ar-run 5 dry`
(or `/ar-run 3 real`) — Claude tunes → runs → scores → repeats itself until N rounds or an
early-stop condition (a promotable round, 3 flat rounds, or an integrity failure). Each per-round
command stays one auditable step; `/ar-run` is the only one that loops on its own. Each command only touches the editable surface; the fail-closed
scorer gates and the integrity guard keep edits honest. The commands are restricted via
`allowed-tools` frontmatter and set `disable-model-invocation: true`, so they run only when you
type them — the model won't fire trials on its own.

### One iteration as a single command

`scripts/ar_loop.py` drives a **single** iteration end to end — trial → score → log → integrity —
and prints one JSON verdict. The slash commands call it; you can also run it directly:

```bash
# dry-run plumbing check (stdlib only, no GPU)
conda run -n boldtembed python scripts/ar_loop.py --dry-run        # or: make autoresearch-loop

# a real iteration on the RTX A6000 (eval-only of the configured model)
conda run -n boldtembed python scripts/ar_loop.py --real --allow-gpu \
  --baseline outputs/autoresearch/baseline/metrics.json \
  --prepared-manifest outputs/autoresearch/prepared/prepare_manifest.json

# a real training trial (checkpoint lands in the run dir, never the promoted path)
conda run -n boldtembed python scripts/ar_loop.py --real --allow-gpu --allow-checkpoints \
  --baseline outputs/autoresearch/baseline/metrics.json \
  --prepared-manifest outputs/autoresearch/prepared/prepare_manifest.json
```

The verdict reports `trial_status`, `score`/`score_status`, `failed_gates`, `leakage_status`,
`integrity`, and a single `promotable` boolean; exit code is 0 **only** when the trial ran, the
score gate passed, and integrity passed. The **outer** loop is the agent's job: read the verdict,
edit `configs/autoresearch/experiments/current.json` (the only editable surface besides the
recipe), and run `ar_loop.py` again. `--real` fails fast with a clear hint if torch is absent (wrong
env). The individual steps are also runnable on their own:

## The loop, end to end (individual steps)

```bash
# 1. (optional) build a preparation manifest from LOCAL data (no downloads)
conda run -n boldtembed python scripts/ar_prepare.py \
  --train data/prepared/train_candidates.jsonl \
  --eval-manifest data/prepared/eval_manifest.json \
  --baseline-model mayflowergmbh/Boldt-Embed-DE-350M \
  --out outputs/autoresearch/prepared \
  --require-leakage-report outputs/autoresearch/leakage_report.json

# 2. run a trial (dry-run shown). For a REAL run add --real --allow-gpu (training also needs
#    --allow-checkpoints), and pass --prepared-manifest so the verified leakage status flows in.
#    A real training trial trains to <run_id>/checkpoint (never the promoted path) and grades THAT.
conda run -n boldtembed python scripts/ar_run_trial.py \
  --config configs/autoresearch/experiments/current.json \
  --budget-minutes 20 \
  --out outputs/autoresearch/runs/<run_id> \
  --baseline outputs/autoresearch/baseline/metrics.json \
  --prepared-manifest outputs/autoresearch/prepared/prepare_manifest.json \
  --dry-run

# 3. score the run against the baseline (canonical, deterministic, fail-closed)
conda run -n boldtembed python scripts/ar_score.py \
  --run outputs/autoresearch/runs/<run_id>/metrics.json \
  --baseline outputs/autoresearch/baseline/metrics.json \
  --out outputs/autoresearch/runs/<run_id>/score.json

# 4. append one auditable row to the results log
conda run -n boldtembed python scripts/ar_log_result.py \
  --run outputs/autoresearch/runs/<run_id> \
  --results outputs/autoresearch/results.tsv
```

## Metrics schema (what the recipe emits and the scorer reads)

```json
{
  "run_id": "...",
  "status": "ok|fail|crash",
  "metrics": {
    "webfaq":     {"recall@100": 0.0, "ndcg@10": 0.0, "mrr@10": 0.0},
    "local_rag":  {"recall@100": 0.0, "ndcg@10": 0.0},
    "germanquad": {"ndcg@10": 0.0},
    "dt_test":    {"ndcg@10": 0.0},
    "matryoshka": {"retention_256": 0.0},
    "leakage":    {"hits": 0, "status": "clean|not_checked|leak_detected|..."},
    "system":     {"vram_gb": 0.0, "throughput_pairs_per_sec": 0.0}
  }
}
```

## Scoring & gates (`scripts/ar_score.py`)

```text
score =
  + 2.0 * Δwebfaq_recall@100
  + 1.5 * Δwebfaq_ndcg@10
  + 1.0 * Δlocal_rag_recall@100   (only if both run and baseline have local_rag)
  + 0.5 * Δwebfaq_mrr@10
  - 3.0 * germanquad_regression_penalty
  - 3.0 * dt_test_regression_penalty
  - 2.0 * matryoshka_256_retention_penalty
  - 0.2 * vram_penalty
  - 0.2 * throughput_penalty
```

`status: "pass"` only if **all** hold: run status ok/pass · **the run is real** (a dry-run /
`scale_disclaimer` trial can never pass — its numbers are plumbing) · **the baseline is a real
measured run** (WebFAQ recall@100 present and > 0, not the 0.0 skeleton) · **leakage is verified
clean** (a missing/`not_checked` status fails closed, not treated as 0) · ΔGermanQuAD nDCG@10 ≥
−0.005 · ΔDT-test nDCG@10 ≥ −0.005 · 256-d retention ≥ 0.95 · WebFAQ recall@100 & nDCG@10 present.

## Real-mode integration

`autoresearch_recipe.run_dense_trial(config, out_dir, deadline_epoch_s, dry_run)` in **real** mode
reuses the existing v6.1 scripts without editing them: it trains via
`scripts/train_v6_1_dense_top50.py` **to `<run_dir>/checkpoint`** (never the promoted canonical
path) and then evaluates **that** checkpoint by calling `eval_v6_1_dense_top50.py`'s own
`dense_eval`/`bm25_eval` functions in-process, so a trial grades exactly the model it produced.
Training is given the budget minus an eval reserve (`runtime.eval_reserve_seconds`, default 300s)
so evaluation can always finish; documents use `max(max_query_length, max_document_length)` as the
sequence length so they are not truncated. Leakage status comes from the prepared manifest. If the
required local data or scripts are missing it returns a clear `status: "fail"` with the missing
integration points listed — it never fabricates metrics. A **real, successful** trial also emits a
canonical run card via `experiment_registry` (`outputs/run-cards/autoresearch-<run_id>.json`) so
the repo's provenance/summary tooling sees it; dry-runs emit nothing.

Note on `system.vram_gb`: evaluation now runs **in-process**, so `torch.cuda.max_memory_allocated()`
reflects the eval peak; a training subprocess's peak is still not captured. Throughput
(`queries_per_sec`) comes from the eval summary and is accurate. The VRAM penalty weight is small
(0.2) and is 0 when run and baseline agree, so it does not distort promotion.
