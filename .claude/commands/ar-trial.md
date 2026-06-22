---
description: Run ONE AutoResearch iteration (trial → score → log → integrity) and report the verdict
argument-hint: "[dry|real]"
allowed-tools: Bash(conda run *) Read
disable-model-invocation: true
---
# AutoResearch — run one iteration

Run a single dense-retriever iteration with the **current** experiment config and report the JSON
verdict. Mode requested: `$ARGUMENTS` (treat empty as `dry`).

- **dry** → plumbing only, no GPU:

  ```bash
  conda run -n boldtembed python scripts/ar_loop.py --dry-run --status keep
  ```

- **real** → real trial on the RTX A6000 via the project env (uses the baseline + prepared manifest
  if present so scoring and the leakage gate are meaningful):

  ```bash
  conda run -n boldtembed python scripts/ar_loop.py --real --allow-gpu --status keep \
    --baseline outputs/autoresearch/baseline/metrics.json \
    --prepared-manifest outputs/autoresearch/prepared/prepare_manifest.json
  ```

Run exactly ONE of these (pick by mode), then summarize the verdict it printed: `run_id`, `mode`,
`trial_status`, `score`, `score_status`, `failed_gates`, `leakage_status`, `integrity`,
`promotable`. 

Rules: do **not** claim any benchmark result beyond what the verdict reports; `dry_run` numbers are
plumbing only and can never be promotable. If `integrity` is `fail` because the AutoResearch files
are still untracked, say so (bootstrap case) rather than treating it as a real violation.
