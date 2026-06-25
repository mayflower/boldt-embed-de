## `README.md`

# Claude Code Prompt Pack: AutoResearch for Boldt-Embed-DE

The implementation target is **dense first-stage retrieval first**. Reranker automation is intentionally postponed until dense recall gates pass.

All experiment trials must use **20-minute training/evaluation budgets** by default:

```bash
python scripts/ar_run_trial.py \
  --config configs/autoresearch/experiments/current.json \
  --budget-minutes 20 \
  --out outputs/autoresearch/runs/<run_id>
```

## Recommended use

Paste prompts into Claude Code in this order:

1. `00-orientation.md`
2. `01-scaffold-files.md`
3. `02-prepare-data-and-manifests.md`
4. `03-score-and-gates.md`
5. `04-run-trial-20min.md`
6. `05-logging-and-run-cards.md`
7. `06-dense-recipe-surface.md`
8. `07-tests-and-validation.md`
9. `08-first-baseline-and-proxy-run.md`
10. `09-reranker-later.md`
11. `10-review-and-merge-checklist.md`

For a single long Claude Code session, use `one-shot.md`.

## Design constraints baked into the prompts

* 20-minute default run budget.
* Stdlib-only core validation where possible.
* Protected eval, leakage, and scoring surfaces.
* Agent may only experiment through config files and `src/boldt_embed/autoresearch_recipe.py`.
* No benchmark claims without saved outputs and run metadata.
* No model weights, datasets, caches, secrets, or HF artifacts committed.
* Dense retriever optimization comes before reranker optimization.

---

## `00-orientation.md`

# Prompt 00 — Orientation and non-negotiables

You are working in the `mayflower/boldt-embed-de` repository.

Your task is to implement an AutoResearch-style experiment loop for Boldt-Embed-DE training. Do not start by changing code. First inspect the repository and summarize the current structure, existing training scripts, evaluation scripts, validation gates, and docs relevant to dense retrieval, reranking, leakage, and release validation.

Important constraints:

1. The first implementation target is **dense retriever AutoResearch**, not reranker AutoResearch.
2. Every actual trial must default to a **20-minute budget**. Use `--budget-minutes 20` as the default. Do not use 30-minute or 5-minute defaults.
3. The AutoResearch loop must not be able to silently edit evaluation data, scoring scripts, leakage checks, baseline outputs, or release gates.
4. Keep the importable core and validation paths stdlib-only unless a script is explicitly a real training/eval path.
5. Do not commit model weights, large datasets, generated caches, secrets, HF downloads, or temporary artifacts.
6. Never claim a benchmark result unless the command was run and output was saved under `outputs/` with run metadata.
7. Use small, reviewable commits or at least small, reviewable patches. Inspect before editing.

Begin by running read-only inspection commands such as:

```bash
git status --short
find . -maxdepth 3 -type f | sort | sed 's#^./##' | head -300
ls scripts configs src/boldt_embed tests docs 2>/dev/null || true
```

Then inspect, if present:

```bash
sed -n '1,220p' README.md
sed -n '1,160p' CLAUDE.md
sed -n '1,220p' docs/v6-dense-rag-and-reranker-plan.md
sed -n '1,220p' docs/dense-recall-gate.md
sed -n '1,220p' docs/data/training-datasets-research-2026.md
sed -n '1,220p' scripts/validate_release_2026.py
sed -n '1,220p' scripts/check_dense_recall_gate.py
sed -n '1,220p' scripts/train_causal.py
sed -n '1,220p' scripts/train_bidirectional.py
sed -n '1,220p' configs/student_training_2026.json
sed -n '1,220p' configs/teacher_models.json
```

After inspection, produce a short implementation plan with:

* Files to add.
* Files to modify.
* Files that must stay protected.
* The validation commands you will run.
* Any uncertainties or missing files discovered during inspection.

Do not implement yet in this prompt. Stop after the plan.

---

## `01-scaffold-files.md`

# Prompt 01 — Add the AutoResearch scaffold

Implement the repository scaffold for a dense-retriever AutoResearch loop.

Create these files if they do not exist:

```text
AUTORESEARCH.md
scripts/ar_prepare.py
scripts/ar_run_trial.py
scripts/ar_score.py
scripts/ar_log_result.py
scripts/check_autoresearch_integrity.py
configs/autoresearch/base_dense.json
configs/autoresearch/experiments/current.json
src/boldt_embed/autoresearch_recipe.py
tests/test_autoresearch_score.py
tests/test_autoresearch_integrity.py
```

Do not wire real GPU training yet. This prompt is for the safe skeleton.

## Requirements for `AUTORESEARCH.md`

Write a concise operating manual for future Claude Code / agentic runs. It must say:

* Goal: improve German dense first-stage retrieval for FAQ/RAG.
* Default run budget: **20 minutes**.
* Editable experiment surface:

  * `configs/autoresearch/experiments/*.json`
  * `src/boldt_embed/autoresearch_recipe.py`
* Protected surfaces:

  * evaluation datasets
  * leakage checks
  * benchmark harnesses
  * scoring scripts
  * release gates
  * baseline outputs
* Hard rules:

  * train != eval
  * leakage hits must be zero
  * GermanQuAD and DT-test are guardrails
  * 256-d Matryoshka retention must be checked
  * no model weights or large datasets in git
  * no benchmark claim without saved run metadata

## Requirements for `configs/autoresearch/base_dense.json`

Create a conservative base config like:

```json
{
  "task": "dense_retriever",
  "base_model": "Boldt/Boldt-DC-350M",
  "reference_model": "mayflowergmbh/Boldt-Embed-DE-350M",
  "budget_minutes": 20,
  "seed": 1337,
  "pooling": "mean",
  "normalize_embeddings": true,
  "matryoshka_dims": [1024, 768, 512, 256, 128, 64],
  "primary_metric": "webfaq_recall_at_100",
  "guardrails": {
    "germanquad_ndcg10_min_delta": -0.005,
    "dt_test_ndcg10_min_delta": -0.005,
    "matryoshka_256_min_retention": 0.95,
    "leakage_hits_max": 0
  },
  "data_mixture": {
    "mmarco_de": 0.45,
    "clips_mqa_de": 0.25,
    "webfaq_train": 0.20,
    "german_stress": 0.10
  },
  "loss": {
    "type": "cached_mnrl_matryoshka_distillation",
    "temperature": 0.03,
    "matryoshka_weight": 1.0,
    "distillation_weight": 0.5,
    "margin_mse_weight": 0.25
  },
  "training": {
    "max_steps": null,
    "budget_minutes": 20,
    "batch_size": 32,
    "grad_accumulation": 1,
    "learning_rate": 0.00002,
    "warmup_ratio": 0.05,
    "max_query_length": 256,
    "max_document_length": 1024,
    "dtype": "bfloat16"
  },
  "runtime": {
    "dry_run": true,
    "allow_gpu": false,
    "write_checkpoints": false
  }
}
```

Make `current.json` initially copy or extend this base config.

## Requirements for script skeletons

Every script should:

* Use `argparse`.
* Have a `main()` returning process exit codes.
* Write JSON with `ensure_ascii=False` and indentation where useful.
* Fail clearly with actionable messages.
* Avoid importing torch/transformers at module import time.
* Be safe to run in stdlib-only mode.

Implement minimal behavior now:

* `ar_prepare.py`: create an output directory and write a manifest stub.
* `ar_run_trial.py`: load a config, enforce `budget_minutes <= 20` unless `--allow-longer-than-20` is explicitly set, create a run directory, and call a dry-run function in `autoresearch_recipe.py`.
* `ar_score.py`: load run metrics and baseline metrics; compute a score using placeholder-safe defaults; fail hard gates when present.
* `ar_log_result.py`: append one row to `outputs/autoresearch/results.tsv`.
* `check_autoresearch_integrity.py`: define protected file globs and fail if a run diff touches them.
* `autoresearch_recipe.py`: expose `run_dense_trial(config, out_dir, deadline_epoch_s, dry_run=True)` that writes deterministic dry-run metrics.

## Validation

Run:

```bash
python scripts/ar_run_trial.py \
  --config configs/autoresearch/experiments/current.json \
  --budget-minutes 20 \
  --out outputs/autoresearch/runs/smoke-dry-run \
  --dry-run

python scripts/ar_score.py \
  --run outputs/autoresearch/runs/smoke-dry-run/metrics.json \
  --baseline outputs/autoresearch/runs/smoke-dry-run/metrics.json \
  --out outputs/autoresearch/runs/smoke-dry-run/score.json

python scripts/ar_log_result.py \
  --run outputs/autoresearch/runs/smoke-dry-run \
  --results outputs/autoresearch/results.tsv

python -m unittest discover -s tests
```

At the end, report:

* Files changed.
* Commands run.
* Validation status.
* Any TODOs left intentionally.

---

## `02-prepare-data-and-manifests.md`

# Prompt 02 — Implement data preparation and manifests

Extend `scripts/ar_prepare.py` into a real stdlib-safe manifest builder for AutoResearch runs.

Do not download datasets. Do not fetch from Hugging Face. This script operates only on local files supplied by the user or produced by existing repository pipelines.

## CLI

Implement:

```bash
python scripts/ar_prepare.py \
  --train data/prepared/train_candidates.jsonl \
  --eval-manifest data/prepared/eval_manifest.json \
  --baseline-model mayflowergmbh/Boldt-Embed-DE-350M \
  --out outputs/autoresearch/prepared
```

Arguments:

```text
--train PATH                     JSONL candidate records
--eval-manifest PATH             JSON eval manifest listing eval sets
--baseline-model STR             reference model id or local path
--out PATH                       output directory
--max-records N                  optional cap for proxy preparation
--seed INT                       default 1337
--require-leakage-report PATH    optional leakage report; fail if missing or hits > 0
--format json|markdown           default json
```

## Expected train JSONL record

Accept records with this shape, allowing extra fields:

```json
{
  "query_id": "...",
  "doc_id": "...",
  "query": "...",
  "document": "...",
  "positive": true,
  "negatives": ["..."],
  "source": "mmarco-de|clips-mqa-de|webfaq-train|synthetic_adversarial",
  "domain": "web|faq|qa|german_stress",
  "license": "...",
  "teacher_scores": {}
}
```

## Expected eval manifest

Support a manifest like:

```json
{
  "sets": [
    {"name": "webfaq", "role": "primary", "path": "data/prepared/eval/webfaq_heldout.jsonl"},
    {"name": "germanquad", "role": "guardrail", "path": "data/prepared/eval/germanquad_guardrail.jsonl"},
    {"name": "dt_test", "role": "guardrail", "path": "data/prepared/eval/dt_test_guardrail.jsonl"},
    {"name": "local_rag", "role": "primary_optional", "path": "data/prepared/eval/local_rag_private.jsonl", "optional": true}
  ]
}
```

## Manifest output

Write:

```text
outputs/autoresearch/prepared/
  prepare_manifest.json
  train_summary.json
  eval_summary.json
```

Include:

* absolute and repo-relative paths where possible
* file size
* SHA256 hash
* record count
* required-field missing counts
* domain/source/license counts for train records
* optional eval sets missing, if any
* baseline model id
* seed
* timestamp UTC
* git commit if available
* leakage report status if provided

## Leakage report handling

If `--require-leakage-report` is provided:

* Load JSON.
* Accept common fields: `hits`, `num_hits`, `leakage_hits`, or nested `summary.hits`.
* Fail if hits > 0.
* Copy the leakage summary into `prepare_manifest.json`.

If no leakage report is provided, mark the preparation as `leakage_status: "not_checked"` and make it clear that promotion is not allowed from this preparation.

## Tests

Add tests for:

* JSONL counting.
* SHA256 hashing.
* missing required fields.
* leakage report hit extraction.
* optional eval set missing does not fail.
* required eval set missing fails.

Run:

```bash
python scripts/ar_prepare.py --help
python -m unittest discover -s tests
```

Report files changed and validation results.

---

## `03-score-and-gates.md`

# Prompt 03 — Implement scoring and hard gates

Implement `scripts/ar_score.py` as the canonical AutoResearch scoring script.

This file is a protected surface after implementation. Future agentic experiments must not edit it except through intentional human review.

## CLI

Support:

```bash
python scripts/ar_score.py \
  --run outputs/autoresearch/runs/<run_id>/metrics.json \
  --baseline outputs/autoresearch/baseline/metrics.json \
  --out outputs/autoresearch/runs/<run_id>/score.json \
  --format json
```

Arguments:

```text
--run PATH
--baseline PATH
--out PATH
--format json|markdown
--germanquad-min-delta FLOAT    default -0.005
--dt-test-min-delta FLOAT       default -0.005
--m256-min-retention FLOAT      default 0.95
--max-leakage-hits INT          default 0
```

## Metrics schema

Support this preferred schema:

```json
{
  "run_id": "...",
  "status": "ok",
  "metrics": {
    "webfaq": {"recall@100": 0.0, "ndcg@10": 0.0, "mrr@10": 0.0},
    "local_rag": {"recall@100": 0.0, "ndcg@10": 0.0},
    "germanquad": {"ndcg@10": 0.0},
    "dt_test": {"ndcg@10": 0.0},
    "matryoshka": {"retention_256": 0.0},
    "leakage": {"hits": 0},
    "system": {"vram_gb": 0.0, "throughput_pairs_per_sec": 0.0}
  }
}
```

Also support common aliases defensively:

```text
recall_at_100, recall100, Recall@100
ndcg_at_10, ndcg10, nDCG@10
mrr_at_10, mrr10, MRR@10
retention256, m256_retention
```

## Score formula

Implement:

```text
score =
  + 2.0 * Δwebfaq_recall@100
  + 1.5 * Δwebfaq_ndcg@10
  + 1.0 * Δlocal_rag_recall@100, if both run and baseline have local_rag
  + 0.5 * Δwebfaq_mrr@10
  - 3.0 * germanquad_regression_penalty
  - 3.0 * dt_test_regression_penalty
  - 2.0 * matryoshka_256_retention_penalty
  - 0.2 * vram_penalty
  - 0.2 * throughput_penalty
```

Define penalties as positive values only:

```text
germanquad_regression_penalty = max(0, germanquad_min_delta - Δgermanquad_ndcg@10)
dt_test_regression_penalty    = max(0, dt_test_min_delta - Δdt_test_ndcg@10)
matryoshka_penalty            = max(0, m256_min_retention - retention_256)
vram_penalty                  = max(0, run_vram_gb - baseline_vram_gb) / max(1, baseline_vram_gb)
throughput_penalty            = max(0, baseline_tput - run_tput) / max(1, baseline_tput)
```

## Hard gates

Return `status: "pass"` only if:

```text
run status is ok/pass
leakage hits <= max_leakage_hits
Δgermanquad_ndcg@10 >= germanquad_min_delta
Δdt_test_ndcg@10 >= dt_test_min_delta
retention_256 >= m256_min_retention
webfaq recall@100 is present
webfaq ndcg@10 is present
```

If a gate fails, include:

```json
"failed_gates": [
  {"name": "matryoshka_256_retention", "value": 0.93, "threshold": 0.95}
]
```

## Output

Write JSON:

```json
{
  "status": "pass|fail",
  "score": 0.0,
  "deltas": {},
  "penalties": {},
  "failed_gates": [],
  "inputs": {"run": "...", "baseline": "..."}
}
```

## Tests

Add tests covering:

* exact score calculation
* alias handling
* no local_rag in either file
* leakage failure
* GermanQuAD guardrail failure
* DT-test guardrail failure
* Matryoshka retention failure
* missing WebFAQ metrics failure

Run:

```bash
python scripts/ar_score.py --help
python -m unittest discover -s tests
```

Report files changed and validation results.

---

## `04-run-trial-20min.md`

# Prompt 04 — Implement the 20-minute trial runner

Implement `scripts/ar_run_trial.py` as the safe trial runner for dense-retriever AutoResearch.

All real trials must default to **20 minutes**. The script must enforce this default and make longer runs impossible unless a human explicitly passes `--allow-longer-than-20`.

## CLI

Support:

```bash
python scripts/ar_run_trial.py \
  --config configs/autoresearch/experiments/current.json \
  --budget-minutes 20 \
  --out outputs/autoresearch/runs/<run_id> \
  --dry-run
```

Arguments:

```text
--config PATH                    required
--out PATH                       required
--budget-minutes INT             default 20
--seed INT                       optional; config seed used if absent
--dry-run                        no GPU, no real training
--real                           allow real training/eval path
--allow-gpu                      allow GPU use
--allow-checkpoints              allow writing checkpoints under output dir
--allow-longer-than-20           explicit escape hatch; still record invalid_for_default_loop=true
--baseline PATH                  optional baseline metrics path
--prepared-manifest PATH         optional data preparation manifest
--notes STR                      optional run notes
```

Rules:

1. If `--budget-minutes` is omitted, use 20.
2. If `--budget-minutes > 20` and `--allow-longer-than-20` is absent, exit nonzero.
3. If `--budget-minutes > 20` and allowed, stamp the run card and metrics with `invalid_for_default_loop: true`.
4. `--real` requires `--allow-gpu` or a config field that explicitly permits CPU real mode.
5. `--allow-checkpoints` may only write inside the run directory or `outputs/autoresearch/checkpoints/<run_id>`.
6. Always pass a deadline timestamp to `autoresearch_recipe.run_dense_trial`.
7. Always save `config.resolved.json`, `command.txt`, `env.json`, `metrics.json`, and `run_card.md`.
8. Always capture exceptions into `error.json` and a failed `metrics.json` rather than leaving an empty run directory.

## Deadline behavior

Compute:

```python
deadline_epoch_s = time.monotonic() + budget_minutes * 60
```

The recipe must be able to check this deadline and stop before exceeding it. The runner should not rely only on shell `timeout`.

## Resolved config

Merge:

1. base config, if `extends` is present
2. experiment config
3. CLI overrides

Implement a small stdlib-only recursive merge helper.

Example `current.json` may contain:

```json
{
  "extends": "configs/autoresearch/base_dense.json",
  "name": "dense_proxy_default_20min",
  "training": {
    "budget_minutes": 20
  },
  "runtime": {
    "dry_run": true,
    "allow_gpu": false,
    "write_checkpoints": false
  }
}
```

## Metrics result

`metrics.json` should include:

```json
{
  "run_id": "...",
  "status": "ok|fail|crash",
  "budget_minutes": 20,
  "elapsed_seconds": 0.0,
  "deadline_respected": true,
  "invalid_for_default_loop": false,
  "config_path": "...",
  "git": {"commit": "...", "dirty": true},
  "metrics": {...}
}
```

## Validation

Run:

```bash
python scripts/ar_run_trial.py --help
python scripts/ar_run_trial.py \
  --config configs/autoresearch/experiments/current.json \
  --budget-minutes 20 \
  --out outputs/autoresearch/runs/trial-runner-smoke \
  --dry-run

# This must fail:
python scripts/ar_run_trial.py \
  --config configs/autoresearch/experiments/current.json \
  --budget-minutes 21 \
  --out outputs/autoresearch/runs/should-fail \
  --dry-run

python -m unittest discover -s tests
```

Report files changed and validation results.

---

## `05-logging-and-run-cards.md`

# Prompt 05 — Implement logging, run cards, and results.tsv

Implement `scripts/ar_log_result.py` and improve run-card generation in `scripts/ar_run_trial.py`.

The AutoResearch loop must be auditable. Every run gets a directory; every scored run gets a TSV row.

## `ar_log_result.py` CLI

Support:

```bash
python scripts/ar_log_result.py \
  --run outputs/autoresearch/runs/<run_id> \
  --results outputs/autoresearch/results.tsv
```

Arguments:

```text
--run PATH                  run directory containing metrics.json and optionally score.json
--results PATH              TSV path; default outputs/autoresearch/results.tsv
--status STR                optional override: keep|discard|crash|invalid_leakage|invalid_guardrail|invalid_for_promotion
--notes STR                 optional notes appended to row
```

## TSV columns

Use stable columns:

```text
timestamp_utc
commit
run_id
status
score
webfaq_recall100
webfaq_ndcg10
webfaq_mrr10
local_rag_recall100
germanquad_ndcg10
dt_test_ndcg10
m256_retention
leakage_hits
budget_minutes
elapsed_seconds
invalid_for_default_loop
vram_gb
throughput_pairs_per_sec
config_path
notes
```

If the TSV does not exist, write the header. If it exists, append. Never rewrite old rows.

## Run card

`ar_run_trial.py` should write `run_card.md` with:

````markdown
# AutoResearch Run: <run_id>

Status: ok|fail|crash
Budget: 20 minutes
Elapsed: ... seconds
Deadline respected: yes/no
Invalid for default loop: yes/no

## Command

```bash
...
```

## Git

- commit: ...
- dirty: true/false
- diff saved: git.diff

## Config

- config path: ...
- resolved config: config.resolved.json

## Metrics

| metric | value |
|---|---:|
| WebFAQ Recall@100 | ... |
| WebFAQ nDCG@10 | ... |
| GermanQuAD nDCG@10 | ... |
| DT-test nDCG@10 | ... |
| Matryoshka 256 retention | ... |

## Score

- score: ...
- failed gates: ...

## Notes

...
````

## Git diff capture

In the runner, capture:

```bash
git diff -- .
git status --short
git rev-parse HEAD
```

Save these under the run directory. This is for audit only; do not fail if git is unavailable.

## Validation

Run a smoke trial, score it, log it:

```bash
rm -rf outputs/autoresearch/runs/logging-smoke
python scripts/ar_run_trial.py \
  --config configs/autoresearch/experiments/current.json \
  --budget-minutes 20 \
  --out outputs/autoresearch/runs/logging-smoke \
  --dry-run
python scripts/ar_score.py \
  --run outputs/autoresearch/runs/logging-smoke/metrics.json \
  --baseline outputs/autoresearch/runs/logging-smoke/metrics.json \
  --out outputs/autoresearch/runs/logging-smoke/score.json
python scripts/ar_log_result.py \
  --run outputs/autoresearch/runs/logging-smoke \
  --results outputs/autoresearch/results.tsv \
  --status discard \
  --notes "smoke test"
python -m unittest discover -s tests
```

Report files changed and validation results.

---

## `06-dense-recipe-surface.md`

# Prompt 06 — Implement the dense recipe experiment surface

Implement `src/boldt_embed/autoresearch_recipe.py` as the only Python file future AutoResearch agents are normally allowed to edit.

The first version should be conservative, clear, and easy to modify safely.

## Public API

Expose:

```python
def run_dense_trial(config: dict, out_dir: str | Path, deadline_epoch_s: float, dry_run: bool = True) -> dict:
    """Run one dense-retriever AutoResearch trial and return metrics dict."""
```

Optional helper APIs:

```python
def validate_recipe_config(config: dict) -> list[str]: ...
def build_training_plan(config: dict) -> dict: ...
def should_stop(deadline_epoch_s: float, reserve_seconds: float = 30.0) -> bool: ...
def write_metrics(out_dir: Path, metrics: dict) -> None: ...
```

## Behavior

### Dry-run mode

Dry-run must:

* require no torch, transformers, sentence-transformers, GPU, datasets, or network
* validate config fields
* create deterministic pseudo-metrics from config and seed
* write `recipe_plan.json`
* return a metrics object matching the scorer schema

Dry-run pseudo-metrics must not look like real benchmark claims. Include:

```json
"scale_disclaimer": "Dry-run pseudo-metrics validate AutoResearch plumbing only; not a benchmark claim."
```

### Real mode

Real mode should initially be a safe adapter, not a full new trainer.

If existing repo scripts for dense training/eval are available, call them through subprocess with explicit command logging. Otherwise fail with a clear message telling the user which integration points are missing.

Real mode must:

* honor the deadline timestamp
* stop before the 20-minute budget expires, leaving enough time to write outputs
* write all temporary outputs inside the run directory
* avoid committing or copying model weights into git-tracked paths
* save real metrics under the same schema as dry-run
* set `scale_disclaimer` honestly for proxy runs

## Training plan fields

Support config fields for future search:

```text
pooling: mean|eos_or_last_token
normalize_embeddings: true|false
matryoshka_dims: list[int]
data_mixture: source -> weight
hard_negatives: bm25|embedder|teacher_vetted|mixed
loss.type: cached_mnrl_matryoshka_distillation|mnrl_only|margin_mse_mix
loss.temperature
loss.matryoshka_weight
loss.distillation_weight
loss.margin_mse_weight
training.learning_rate
training.warmup_ratio
training.batch_size
training.grad_accumulation
training.max_query_length
training.max_document_length
lora.rank
lora.alpha
lora.target_modules
```

Do not require all fields. Apply defaults.

## Initial real-mode command integration

Look for existing scripts in the repo and integrate only if they exist. Candidate commands:

```bash
python scripts/train_causal.py --config <generated-config> --data <prepared-data> --out <run-checkpoint>
python scripts/train_bidirectional.py --config <generated-config>
python scripts/run_mteb_benchmark_template.py --model <checkpoint> --config <eval-config>
```

If the repository has a v6 dense training/eval script, prefer that. If not, leave a documented `NotImplementedError` for real mode while keeping dry-run and tests passing.

## Guardrails

Do not put scoring logic in this recipe file. The canonical score remains `scripts/ar_score.py`.

Do not put leakage logic in this recipe file. It may read the preparation manifest and include leakage status, but leakage checking remains protected.

## Validation

Run:

```bash
python scripts/ar_run_trial.py \
  --config configs/autoresearch/experiments/current.json \
  --budget-minutes 20 \
  --out outputs/autoresearch/runs/recipe-smoke \
  --dry-run
python -m unittest discover -s tests
```

Report files changed, commands run, validation status, and remaining real-mode integration TODOs.

---

## `07-tests-and-validation.md`

# Prompt 07 — Add tests and repository validation wiring

Add robust unit tests for the AutoResearch implementation and wire the new scripts into existing validation where appropriate.

## Tests to add or extend

Create or update:

```text
tests/test_autoresearch_prepare.py
tests/test_autoresearch_score.py
tests/test_autoresearch_trial_runner.py
tests/test_autoresearch_log_result.py
tests/test_autoresearch_integrity.py
tests/test_autoresearch_recipe.py
```

Tests should use only stdlib:

```python
import json
import tempfile
import unittest
from pathlib import Path
```

Do not require torch, transformers, sentence-transformers, datasets, network, GPU, or large files.

## Required coverage

### Prepare

* JSONL count.
* Required-field validation.
* Source/domain/license counts.
* SHA256 hash calculation.
* optional eval set missing allowed.
* required eval set missing fails.
* leakage report hit extraction.
* leakage report with hits fails when required.

### Score

* exact weighted score.
* aliases for metric names.
* pass when gates pass.
* fail on leakage.
* fail on GermanQuAD regression beyond threshold.
* fail on DT-test regression beyond threshold.
* fail on 256-d retention below threshold.
* fail on missing WebFAQ metrics.

### Trial runner

* default budget is 20.
* budget 21 fails without `--allow-longer-than-20`.
* budget 21 passes with the escape hatch and stamps `invalid_for_default_loop`.
* dry-run writes required output files.
* crash path writes `error.json` and failed metrics.

### Log result

* writes header on first row.
* appends second row without changing header.
* handles missing score file.
* stores notes.

### Integrity

* protected glob detection works.
* allowed experiment files pass.
* scoring script edits fail.
* eval file edits fail.

### Recipe

* dry-run deterministic with same seed.
* dry-run changes when relevant config changes.
* metrics schema compatible with scorer.
* deadline helper returns true near deadline.

## Validation commands

Run:

```bash
python -m unittest discover -s tests
python scripts/validate_repo.py --format markdown || true
python scripts/run_smoke_tests.py --format markdown || true
python scripts/run_local_benchmark.py --format markdown || true
```

If existing validation scripts fail for reasons unrelated to AutoResearch, report them honestly with the relevant error excerpts. Do not hide failures.

## Optional Makefile wiring

If the repo has a `Makefile`, add a non-invasive target:

```make
.PHONY: autoresearch-smoke
autoresearch-smoke:
	python scripts/ar_run_trial.py --config configs/autoresearch/experiments/current.json --budget-minutes 20 --out outputs/autoresearch/runs/make-smoke --dry-run
	python scripts/ar_score.py --run outputs/autoresearch/runs/make-smoke/metrics.json --baseline outputs/autoresearch/runs/make-smoke/metrics.json --out outputs/autoresearch/runs/make-smoke/score.json
	python scripts/ar_log_result.py --run outputs/autoresearch/runs/make-smoke --status discard --notes "make smoke"
```

Do not make this target part of default `make all` unless the repository maintainer asks.

Report files changed and validation results.

---

## `08-first-baseline-and-proxy-run.md`

# Prompt 08 — Create baseline and run the first 20-minute proxy trial

Use this prompt only after the scaffold, scoring, runner, logging, and tests are in place.

Goal: create the first auditable AutoResearch baseline and run one proxy trial with a **20-minute budget**.

## Step 1: inspect available data and eval files

Run read-only discovery:

```bash
find data benchmarks outputs -maxdepth 4 -type f 2>/dev/null | sort | sed -n '1,240p'
find configs -maxdepth 4 -type f | sort
find scripts -maxdepth 2 -type f | sort | grep -E 'eval|bench|dense|recall|mteb|v6|train' || true
```

Identify whether these exist locally:

```text
data/prepared/train_candidates.jsonl
data/prepared/eval_manifest.json
outputs/v6-dense-rag/dense_recall_gate.json
outputs/v6-dense-rag/webfaq_real_recall_bm25_vs_dense.json
outputs/run-cards/*
```

Do not download anything unless the user explicitly approves.

## Step 2: create a baseline metrics file

If a trusted existing dense baseline metrics JSON exists, adapt it into:

```text
outputs/autoresearch/baseline/metrics.json
```

If no trusted baseline exists, create a dry-run baseline only and mark it clearly:

```json
{
  "run_id": "baseline-dry-run",
  "status": "ok",
  "scale_disclaimer": "Dry-run baseline only; not a benchmark claim.",
  "metrics": {
    "webfaq": {"recall@100": 0.0, "ndcg@10": 0.0, "mrr@10": 0.0},
    "germanquad": {"ndcg@10": 0.0},
    "dt_test": {"ndcg@10": 0.0},
    "matryoshka": {"retention_256": 1.0},
    "leakage": {"hits": 0},
    "system": {"vram_gb": 0.0, "throughput_pairs_per_sec": 0.0}
  }
}
```

Do not pretend dry-run metrics are model quality.

## Step 3: run preparation if local files exist

If local prepared data exists:

```bash
python scripts/ar_prepare.py \
  --train data/prepared/train_candidates.jsonl \
  --eval-manifest data/prepared/eval_manifest.json \
  --baseline-model mayflowergmbh/Boldt-Embed-DE-350M \
  --out outputs/autoresearch/prepared \
  --require-leakage-report outputs/autoresearch/leakage_report.json
```

If no leakage report exists, run without `--require-leakage-report` and mark the preparation not promotable.

## Step 4: run the first proxy trial

Use exactly 20 minutes as the configured budget. For dry-run smoke:

```bash
python scripts/ar_run_trial.py \
  --config configs/autoresearch/experiments/current.json \
  --budget-minutes 20 \
  --out outputs/autoresearch/runs/first-proxy-20min \
  --baseline outputs/autoresearch/baseline/metrics.json \
  --dry-run \
  --notes "first AutoResearch proxy dry-run; not a benchmark claim"
```

For real mode, only run if the environment is already configured and the user has approved GPU usage:

```bash
python scripts/ar_run_trial.py \
  --config configs/autoresearch/experiments/current.json \
  --budget-minutes 20 \
  --out outputs/autoresearch/runs/first-real-20min \
  --baseline outputs/autoresearch/baseline/metrics.json \
  --prepared-manifest outputs/autoresearch/prepared/prepare_manifest.json \
  --real \
  --allow-gpu \
  --allow-checkpoints \
  --notes "first 20-minute real proxy trial"
```

## Step 5: score and log

```bash
python scripts/ar_score.py \
  --run outputs/autoresearch/runs/first-proxy-20min/metrics.json \
  --baseline outputs/autoresearch/baseline/metrics.json \
  --out outputs/autoresearch/runs/first-proxy-20min/score.json

python scripts/ar_log_result.py \
  --run outputs/autoresearch/runs/first-proxy-20min \
  --results outputs/autoresearch/results.tsv \
  --status discard \
  --notes "first proxy plumbing run"
```

## Step 6: summarize honestly

Report:

* Commands run.
* Whether the run was dry-run or real.
* Whether metrics are benchmark claims or plumbing only.
* Score status and failed gates.
* Where outputs were saved.
* What the next real experiment should change.

---

## `09-reranker-later.md`

# Prompt 09 — Reranker AutoResearch, later only

Do not implement this until dense first-stage recall is good enough and the user asks for reranker automation.

When dense recall passes the recall sufficiency gate, create a separate reranker AutoResearch loop. It must not share the dense scoring script except for common utilities.

## Files to add later

```text
AUTORESEARCH_RERANKER.md
configs/autoresearch/base_reranker.json
configs/autoresearch/reranker/current.json
src/boldt_embed/autoresearch_reranker_recipe.py
scripts/ar_run_reranker_trial.py
scripts/ar_score_reranker.py
tests/test_autoresearch_reranker_score.py
```

## Key rule

The reranker must be evaluated as **raw lift over fixed candidate lists**. No abstain policy, bounded policy, margin override serving wrapper, or other policy-gated workaround counts as promotion evidence.

## Reranker default budget

Use **20 minutes** by default, just like dense trials:

```bash
python scripts/ar_run_reranker_trial.py \
  --config configs/autoresearch/reranker/current.json \
  --budget-minutes 20 \
  --out outputs/autoresearch/reranker-runs/<run_id>
```

## Hard gates

Reranker trials can only be promotable if:

```text
dense recall STOP file is absent
candidate lists are fixed before reranker training
WebFAQ/local RAG raw nDCG@10 lift clears threshold
GermanQuAD/DT-test do not regress beyond tolerance
catastrophic-drop rate is within tolerance
leakage hits are zero
policy-gated results are ignored for promotion
```

## Allowed search space

Allow the agent to vary:

```text
candidate-list composition
teacher-score temperature
listwise KL vs pairwise BCE
rank-preservation penalty
negative sampling by rank band
query/document template
max length
LoRA rank and target modules
learning rate and warmup
```

## Protected surfaces

Protect:

```text
fixed candidate lists
eval harness
reranker scoring script
release validation gates
leakage checks
baseline outputs
```

## Instruction for Claude Code when this phase starts

First inspect the dense recall gate outputs and confirm that positives are present often enough. If recall sufficiency is not met, stop and report that reranker AutoResearch would waste GPU because the candidate lists do not contain the positives.

---

## `10-review-and-merge-checklist.md`

# Prompt 10 — Review and merge checklist

Use this prompt after implementation is complete.

Perform a final review of the AutoResearch implementation.

## Required checks

Run:

```bash
git status --short
python -m unittest discover -s tests
python scripts/ar_run_trial.py \
  --config configs/autoresearch/experiments/current.json \
  --budget-minutes 20 \
  --out outputs/autoresearch/runs/final-review-smoke \
  --dry-run
python scripts/ar_score.py \
  --run outputs/autoresearch/runs/final-review-smoke/metrics.json \
  --baseline outputs/autoresearch/runs/final-review-smoke/metrics.json \
  --out outputs/autoresearch/runs/final-review-smoke/score.json
python scripts/ar_log_result.py \
  --run outputs/autoresearch/runs/final-review-smoke \
  --status discard \
  --notes "final review smoke"
python scripts/check_autoresearch_integrity.py --help
```

Also run existing validation if available:

```bash
python scripts/validate_repo.py --format markdown || true
python scripts/run_smoke_tests.py --format markdown || true
python scripts/run_local_benchmark.py --format markdown || true
```

## Review points

Confirm:

* Default trial budget is exactly 20 minutes.
* Runs over 20 minutes fail unless `--allow-longer-than-20` is present.
* Longer runs are marked invalid for the default AutoResearch loop.
* Dry-run mode needs no torch, transformers, datasets, network, or GPU.
* Scoring script is deterministic and tested.
* Guardrails fail closed.
* Leakage status is visible in manifests and score gates.
* Eval/scoring/leakage/release-gate surfaces are protected.
* Result logs append rather than overwrite.
* Run cards include command, commit, config, metrics, score, and notes.
* No model weights, datasets, HF caches, checkpoints, or secrets are staged.
* `.gitignore` covers `outputs/autoresearch/runs/`, `outputs/autoresearch/checkpoints/`, and other generated heavy artifacts if necessary.
* `AUTORESEARCH.md` tells future agents exactly what they can edit.

## Final response format

Report:

```text
Files changed
Commands run
Validation results
Known limitations
Next recommended experiment
Risk notes
```

Be explicit about whether anything is a dry-run plumbing check versus a real benchmark claim.

---

## `one-shot.md`

# One-shot Claude Code prompt: implement AutoResearch for Boldt-Embed-DE with 20-minute runs

You are working in the `mayflower/boldt-embed-de` repository.

Implement an AutoResearch-style dense-retriever experiment loop. Use **20-minute runs by default**, not 5-minute or 30-minute runs.

## First inspect

Before editing, inspect relevant files:

```bash
git status --short
find . -maxdepth 3 -type f | sort | sed 's#^./##' | head -300
sed -n '1,180p' README.md
sed -n '1,140p' CLAUDE.md
sed -n '1,220p' docs/v6-dense-rag-and-reranker-plan.md 2>/dev/null || true
sed -n '1,220p' docs/dense-recall-gate.md 2>/dev/null || true
sed -n '1,220p' docs/data/training-datasets-research-2026.md 2>/dev/null || true
sed -n '1,220p' scripts/check_dense_recall_gate.py 2>/dev/null || true
sed -n '1,220p' scripts/validate_release_2026.py 2>/dev/null || true
```

Then implement small, reviewable patches.

## Files to add

```text
AUTORESEARCH.md
scripts/ar_prepare.py
scripts/ar_run_trial.py
scripts/ar_score.py
scripts/ar_log_result.py
scripts/check_autoresearch_integrity.py
configs/autoresearch/base_dense.json
configs/autoresearch/experiments/current.json
src/boldt_embed/autoresearch_recipe.py
tests/test_autoresearch_prepare.py
tests/test_autoresearch_score.py
tests/test_autoresearch_trial_runner.py
tests/test_autoresearch_log_result.py
tests/test_autoresearch_integrity.py
tests/test_autoresearch_recipe.py
```

## Core requirements

1. Implement dense retriever AutoResearch first. Do not implement reranker automation yet.
2. Default trial budget is exactly **20 minutes**:

```bash
python scripts/ar_run_trial.py \
  --config configs/autoresearch/experiments/current.json \
  --budget-minutes 20 \
  --out outputs/autoresearch/runs/<run_id>
```

3. If `--budget-minutes > 20`, fail unless `--allow-longer-than-20` is passed.
4. If longer runs are allowed, stamp them with `invalid_for_default_loop: true`.
5. Do not import torch/transformers at module import time.
6. Dry-run mode must require no torch, transformers, datasets, network, GPU, or large files.
7. Real mode may initially be a safe adapter to existing training/eval scripts or a clear `NotImplementedError`, but dry-run and tests must pass.
8. Scoring and gates must be deterministic and stdlib-only.
9. Do not commit model weights, datasets, checkpoints, HF caches, or secrets.
10. Never claim a benchmark unless the output is saved under `outputs/` with run metadata.

## Protected surfaces

Future AutoResearch agents may edit only:

```text
configs/autoresearch/experiments/*.json
src/boldt_embed/autoresearch_recipe.py
```

Protect:

```text
evaluation data
leakage checks
benchmark harnesses
scoring scripts
release gates
baseline outputs
```

Implement `scripts/check_autoresearch_integrity.py` to detect protected-surface edits from git diff/status paths.

## `ar_prepare.py`

Implement a stdlib-safe manifest builder.

CLI:

```bash
python scripts/ar_prepare.py \
  --train data/prepared/train_candidates.jsonl \
  --eval-manifest data/prepared/eval_manifest.json \
  --baseline-model mayflowergmbh/Boldt-Embed-DE-350M \
  --out outputs/autoresearch/prepared
```

It should write:

```text
prepare_manifest.json
train_summary.json
eval_summary.json
```

Include file paths, SHA256, sizes, record counts, required-field missing counts, source/domain/license counts, baseline model, timestamp UTC, git commit, and leakage report status if provided.

If `--require-leakage-report` is provided, fail when leakage hits > 0. Accept fields `hits`, `num_hits`, `leakage_hits`, or `summary.hits`.

## `ar_run_trial.py`

Implement a safe runner that:

* merges config via `extends`
* enforces 20-minute default budget
* computes `deadline_epoch_s = time.monotonic() + budget_minutes * 60`
* calls `src/boldt_embed/autoresearch_recipe.py::run_dense_trial`
* writes `config.resolved.json`, `command.txt`, `env.json`, `git.diff`, `git.status`, `metrics.json`, and `run_card.md`
* writes `error.json` and failed metrics on crash
* never writes checkpoints outside allowed output paths

## `autoresearch_recipe.py`

Expose:

```python
def run_dense_trial(config: dict, out_dir, deadline_epoch_s: float, dry_run: bool = True) -> dict:
    ...
```

Dry-run should validate config, produce deterministic pseudo-metrics from config + seed, write `recipe_plan.json`, and include:

```json
"scale_disclaimer": "Dry-run pseudo-metrics validate AutoResearch plumbing only; not a benchmark claim."
```

Real mode should either call existing repo scripts safely or fail clearly with a TODO. It must honor the 20-minute deadline.

## `ar_score.py`

Support:

```bash
python scripts/ar_score.py \
  --run outputs/autoresearch/runs/<run_id>/metrics.json \
  --baseline outputs/autoresearch/baseline/metrics.json \
  --out outputs/autoresearch/runs/<run_id>/score.json
```

Preferred metrics schema:

```json
{
  "run_id": "...",
  "status": "ok",
  "metrics": {
    "webfaq": {"recall@100": 0.0, "ndcg@10": 0.0, "mrr@10": 0.0},
    "local_rag": {"recall@100": 0.0, "ndcg@10": 0.0},
    "germanquad": {"ndcg@10": 0.0},
    "dt_test": {"ndcg@10": 0.0},
    "matryoshka": {"retention_256": 0.0},
    "leakage": {"hits": 0},
    "system": {"vram_gb": 0.0, "throughput_pairs_per_sec": 0.0}
  }
}
```

Score formula:

```text
score =
  + 2.0 * Δwebfaq_recall@100
  + 1.5 * Δwebfaq_ndcg@10
  + 1.0 * Δlocal_rag_recall@100, if both files have local_rag
  + 0.5 * Δwebfaq_mrr@10
  - 3.0 * germanquad_regression_penalty
  - 3.0 * dt_test_regression_penalty
  - 2.0 * matryoshka_256_retention_penalty
  - 0.2 * vram_penalty
  - 0.2 * throughput_penalty
```

Hard gates:

```text
run status ok/pass
leakage hits <= 0
Δgermanquad_ndcg@10 >= -0.005
Δdt_test_ndcg@10 >= -0.005
retention_256 >= 0.95
webfaq recall@100 present
webfaq ndcg@10 present
```

## `ar_log_result.py`

Append one TSV row to `outputs/autoresearch/results.tsv` with stable columns:

```text
timestamp_utc commit run_id status score webfaq_recall100 webfaq_ndcg10 webfaq_mrr10 local_rag_recall100 germanquad_ndcg10 dt_test_ndcg10 m256_retention leakage_hits budget_minutes elapsed_seconds invalid_for_default_loop vram_gb throughput_pairs_per_sec config_path notes
```

Write a header if the file does not exist. Never rewrite old rows.

## Tests

Use only stdlib `unittest`, `tempfile`, `json`, and `pathlib` where possible.

Add tests for:

* 20-minute default budget.
* 21-minute budget fails without override.
* 21-minute budget with override is marked invalid for default loop.
* scoring formula and metric aliases.
* leakage, GermanQuAD, DT-test, Matryoshka, and missing WebFAQ failures.
* prepare JSONL counts, hashes, and leakage extraction.
* logging header and append behavior.
* protected-file detection.
* deterministic recipe dry-run.

## Validation commands

Run:

```bash
python scripts/ar_run_trial.py --help
python scripts/ar_score.py --help
python scripts/ar_prepare.py --help
python scripts/ar_log_result.py --help
python scripts/check_autoresearch_integrity.py --help
python scripts/ar_run_trial.py \
  --config configs/autoresearch/experiments/current.json \
  --budget-minutes 20 \
  --out outputs/autoresearch/runs/one-shot-smoke \
  --dry-run
python scripts/ar_score.py \
  --run outputs/autoresearch/runs/one-shot-smoke/metrics.json \
  --baseline outputs/autoresearch/runs/one-shot-smoke/metrics.json \
  --out outputs/autoresearch/runs/one-shot-smoke/score.json
python scripts/ar_log_result.py \
  --run outputs/autoresearch/runs/one-shot-smoke \
  --status discard \
  --notes "one-shot smoke"
python -m unittest discover -s tests
```

Also run existing validations if present, but do not hide unrelated failures:

```bash
python scripts/validate_repo.py --format markdown || true
python scripts/run_smoke_tests.py --format markdown || true
python scripts/run_local_benchmark.py --format markdown || true
```

## Final response format

Report:

```text
Files changed
Commands run
Validation results
Benchmark status: dry-run plumbing only or real metrics
Known limitations
Next recommended experiment
Risks
```

