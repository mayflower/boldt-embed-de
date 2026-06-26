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

There is **ONE loop** — `/ar-run` — and a few read-only/single-step helpers. You (Claude Code) are
the orchestrator: `/ar-run` reads the measured state off disk, picks the single most reasonable next
experiment, runs it, judges it, and repeats. There is **no controller, no state file, no fixed step
order.**

| Command | What it does |
|---|---|
| `/ar-run [rounds] [dry\|real]` | **THE loop.** Each round: assess state → pick the best next lever (tune / data / specialist / merge / distill) → run → eval on MTEB(deu) + gate → keep-or-revert. |
| `/ar-orient` | Brief on the goal, rules, editable vs protected surfaces, integrity status |
| `/ar-status` | Summarize `results.tsv` + the current config; best WebFAQ recall so far |
| `/ar-prepare <args>` | Build the preparation manifest from local data (+ leakage report) |
| `/ar-trial [dry\|real]` | Run ONE proxy iteration (trial → score → log → integrity); report the verdict |
| `/ar-tune [hypothesis]` | Make ONE editable-surface config change toward WebFAQ recall, then iterate |
| `/ar-integrity [--base-ref REF]` | Run the protected-surface check and explain any violations |

`/ar-run` is the only thing that loops; the rest are single-shot helpers for when you want to do one
step or just look. Commands set `disable-model-invocation: true`, so they run only when you type them.

## What `/ar-run` is optimizing (and the levers it picks from)

**Goal:** beat the **same-size peers** (`multilingual-e5-base` 278M, `LFM2.5-Embedding-350M`) on the
official **MTEB(deu) retrieval-core** — GermanQuAD-Retrieval, GerDaLIRSmall,
MIRACLRetrievalHardNegatives, MultiLongDocRetrieval (nDCG@10, German subset). Qwen3-Embedding-0.6B is
a *stretch reference*, not a gate (it is larger). Promotion is graded by
`scripts/check_mteb_frontier_gate.py` (via `scripts/ar_promote.py`) on the saved
`outputs/mteb/<label>/summary.json` — never claimed beyond the saved file. The in-loop WebFAQ proxy
is a cheap inner signal `/ar-run` may use to triage cheap knob changes.

**What's established (so the loop doesn't relitigate dead ends):**
- The lever that moves every task is **DATA** — real fetched online corpora (SWIM-IR moved MIRACL
  0.332→0.385). Architecture changes did not: **bidirectional / LLM2Vec is a net loss** (clean A/B
  causal beat it 3/4, even with MNTP); **more steps saturate**; **SLERP of near-identical
  checkpoints is a no-op**; **local recombination just dilutes**.
- No single ~1M mix maxes all four tasks — the **composition trade-off**: pure-wiki (SWIM-IR) maxes
  MIRACL/GermanQuAD; web (mMARCO) + FAQ (mqa) maxes GerDaLIR but dilutes MIRACL.
- The escape: **balanced data → complementary specialists → merge → listwise-KL distill**.

**The lever menu** `/ar-run` chooses from each round (it calls these scripts; you don't run them in a
fixed order):

| Lever | Script | When the state says to pick it |
|---|---|---|
| tune | `scripts/ar_loop.py` | cheap knob hill-climb on the proxy |
| data | `scripts/ar_build_mixture.py` | composition is the limiter (build a balanced clean mix) |
| specialist | `scripts/ar_train_specialist.py` (+ `ar_refresh_hardnegatives.py`) | build a complementary domain expert from the shared warm-start |
| merge | `scripts/ar_merge_search.py` | ≥2 complementary checkpoints exist — the trade-off escape (cheapest high-value) |
| distill | `scripts/ar_distill_trial.py` | sharpen ranking from the Qwen3-Reranker teacher |
| measure | `scripts/ar_mteb_trial.py` + `scripts/ar_promote.py` | eval + gate any produced checkpoint |

State is read from the artifacts on disk — `scripts/ar_frontier_status.py` (ranked candidates, peer
frontier, per-task leaders = complementary merge inputs) and `scripts/ar_report.py` (Pareto view).
**The data-source catalogue** `configs/data_sources.json` is the single source of truth for what may
train: only `training_usable:true` **and** `leakage ∈ {scanned_clean, clean}` (fail-closed
otherwise). The mixture builder (`data_mixture_optimizer.build_mixture`, which the recipe's
`materialize_mixture` path also delegates to) is the one place that selects/dedups/FAQ-caps, so the
gate can't drift. Hygiene the loop keeps: eval all candidates at the **same seq length** for a fair
peer comparison; merge only specialists that share the warm-start basin; never train on a source
that isn't scanned-clean.

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
so evaluation can always finish. The sequence length is `max(max_query_length, max_document_length)`
but **capped so `batch_size × seq_length` stays at the v6.1-proven memory point (256 × 32)** — at the
default batch 32 a 1024-token document request is capped to 256 (32 × 1024 OOMs the 48 GB A6000), and
a config needing longer documents must lower `batch_size` to buy the length (the cap is recorded in
the plan as `max_seq_length_requested` / `seq_capped_for_memory`, never silent). The recipe forwards
the tunable knobs to the trainer: `learning_rate`, `warmup_ratio`, `loss.temperature`
(→ CMNRL `scale = 1/temperature`), `batch_size` (= the contrastive in-batch-negative count),
`max_seq_length`, and `dtype`/`bf16` — so tuning these in `current.json` changes a **real** run, not
just the dry-run plan. When a verified-clean prepared manifest is supplied, real training uses the
manifest's **certified cleaned** file (fail-closed if it is missing), so the leakage gate's status
and the data actually trained on are provably the same file. Data-distribution knobs (`data_mixture`,
hard negatives) are realized in the `train_pairs`/`hard_negatives` files and must pass the leakage
scan; `pooling`/`matryoshka_dims` remain plan-only for now. Leakage status comes from the prepared manifest. If the
required local data or scripts are missing it returns a clear `status: "fail"` with the missing
integration points listed — it never fabricates metrics. A **real, successful** trial also emits a
canonical run card via `experiment_registry` (`outputs/run-cards/autoresearch-<run_id>.json`) so
the repo's provenance/summary tooling sees it; dry-runs emit nothing.

Note on `system.vram_gb`: evaluation now runs **in-process**, so `torch.cuda.max_memory_allocated()`
reflects the eval peak; a training subprocess's peak is still not captured. Throughput
(`queries_per_sec`) comes from the eval summary and is accurate. The VRAM penalty weight is small
(0.2) and is 0 when run and baseline agree, so it does not distort promotion.
