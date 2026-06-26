---
description: AutoResearch state-machine controller — status / next / plan a trial (never starts GPU)
argument-hint: "[status | next | plan --trial-type TYPE] [--real --allow-gpu]"
allowed-tools: Bash(conda run *) Bash(cat *) Bash(tail *) Read
disable-model-invocation: true
---
# AutoResearch Controller

The conservative brain of the v8 program: it reads `outputs/autoresearch/events.jsonl`, decides the
next trial with a deterministic ladder, and PRINTS the command to run. It never starts GPU/teacher
work itself. Default: dry-run / plan only.

Parse `$ARGUMENTS` (default `status`):
- `status` → `conda run -n boldtembed python scripts/ar_controller.py status`
- `next` → `conda run -n boldtembed python scripts/ar_controller.py next --dry-run`
- `plan --trial-type TYPE` → `conda run -n boldtembed python scripts/ar_controller.py plan --trial-type TYPE --dry-run`
  (TYPE ∈ data_mix, dense, hardneg_refresh, specialist, merge, distill, mteb, promotion)

Report: the decided trial type, the exact command it planned, expected output artifacts, and the
recommended next step. To execute a planned real step, run that command yourself with its explicit
`--real`/`--allow-*` flags. After a step completes, record it:
`conda run -n boldtembed python scripts/ar_controller.py record --event-json <event.json>`.
