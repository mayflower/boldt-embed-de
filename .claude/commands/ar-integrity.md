---
description: Run the AutoResearch protected-surface integrity check and explain any violations
argument-hint: "[--base-ref REF]"
allowed-tools: Bash(conda run *)
disable-model-invocation: true
---
# AutoResearch — integrity check

Confirm that only the editable surface (`configs/autoresearch/experiments/*.json` and the recipe)
changed — nothing in scoring, gates, eval data, leakage checks, baselines, or the base config.

```bash
conda run -n boldtembed python scripts/check_autoresearch_integrity.py $ARGUMENTS --format json
```

Report `status` and list any `violations`, explaining what each protected file is. Notes:
- If the AutoResearch scripts are still **untracked** (not yet committed), they will appear as
  violations — that is the bootstrap case, not a loop violation. After committing them, pass
  `--base-ref <that-commit>` so the check vets changes *since* the loop started (this also catches a
  protected edit that was committed rather than left in the working tree).
- A real promotion requires this check to pass.
