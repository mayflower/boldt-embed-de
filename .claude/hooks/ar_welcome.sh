#!/usr/bin/env bash
# SessionStart hook: greet the user with a short AutoResearch intro.
# Emits {"systemMessage": "..."} so Claude Code shows the text to the user at session start.
# Python (stdlib json) handles the escaping so the message stays readable here.
python3 - <<'PY'
import json
msg = """\
🔬 AutoResearch loop — German dense retriever (Boldt-Embed-DE)

This repo is instrumented for an iterative dense first-stage retrieval experiment. Drive it with
these slash commands (type them at the prompt):

  /ar-orient     rules, editable vs protected surfaces, current integrity status
  /ar-status     recent results + current config; best WebFAQ recall so far
  /ar-prepare    build the data + leakage manifest from local files
  /ar-trial      run ONE iteration (trial → score → log → integrity); add "real" for the A6000
  /ar-tune       change one knob toward WebFAQ recall, then iterate
  /ar-run        run MANY rounds autonomously in one go (e.g. /ar-run 5 dry)
  /ar-integrity  verify only the editable surface changed

The loop: edit configs/autoresearch/experiments/current.json → /ar-trial → read the verdict →
repeat — or hand it to /ar-run to do several rounds hands-free. WebFAQ recall@100 is the PRIMARY metric; GermanQuAD/DT-test are do-not-regress guardrails;
dry-run numbers are plumbing only. Real trials use the `boldtembed` conda env on the RTX A6000.

New here? Start with /ar-orient. Full rules: AUTORESEARCH.md.\
"""
print(json.dumps({"systemMessage": msg}))
PY
