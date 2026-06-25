---
description: Compose a leakage-clean training mixture from the data-source catalogue (for the next /ar-run)
argument-hint: "<src1:w1,src2:w2,...> [total]"
allowed-tools: Bash(conda run *) Bash(cat *) Read Edit
disable-model-invocation: true
---
# AutoResearch — compose a training data mixture

Set up a REAL training mixture from `configs/data_sources.json` so the next `/ar-run real` trains on
it. Parse `$ARGUMENTS`: a comma list of `source_id:weight` (weights are normalized); optional final
token = total rows (default 500000). Empty args → just show the catalogue.

Only `training_usable:true` AND `leakage:"scanned_clean"` sources may be mixed — the recipe is
FAIL-CLOSED on anything else (the union of individually-scanned-clean sources is clean by
construction, so no slow re-scan). FAQ is auto-capped (faq_cap=0.30) + domain-balanced.

1. Show the usable catalogue sources:
   ```bash
   conda run -n boldtembed python -c "import json;d=json.load(open('configs/data_sources.json'));[print(s['id'],s.get('rows') or s.get('rows_clean'),s['domain'],s['leakage']) for g in ('train_pairs_processed_unions','train_pairs_raw_sources') for s in d[g] if s.get('training_usable') and s.get('leakage') in ('scanned_clean','clean')]"
   ```
2. Edit ONLY `configs/autoresearch/experiments/current.json` (the editable surface): set
   `data_mixture` to the requested `{source_id: weight}`, and `runtime.materialize_mixture=true`,
   `runtime.mixture_total=<total>`, `runtime.faq_cap=0.30`. State the one-line rationale for the mix.
3. Dry-run-verify the mixture materializes clean (builds a tiny sample, fail-closed check, no GPU):
   ```bash
   PYTHONPATH=src conda run -n boldtembed python -c "import json,pathlib,tempfile;import boldt_embed.autoresearch_recipe as r;cfg=json.load(open('configs/autoresearch/experiments/current.json'));cfg['runtime']['mixture_total']=400;[print(*r._materialize_data_mixture(cfg,pathlib.Path(tempfile.mkdtemp()),[]))]"
   ```
4. Report the planned mix (sources + weights + budgets + FAQ balance), confirm all sources are
   scanned_clean, and tell me to run `/ar-run 1 real` to train on it. Do NOT start training here.
