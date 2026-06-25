---
description: Orient on the AutoResearch loop — goal, rules, editable vs protected surfaces, next step
argument-hint: ""
allowed-tools: Bash(conda run *) Read
disable-model-invocation: true
---
# AutoResearch — orientation

Read `AUTORESEARCH.md` in the repo root, then brief me in a few bullets on:

- the **goal** (improve German dense first-stage retrieval; WebFAQ recall@100 is the PRIMARY metric;
  GermanQuAD / DT-test are do-not-regress **guardrails**, never the primary signal),
- the **20-minute** per-trial budget,
- what I may edit — **only** `configs/autoresearch/experiments/current.json` and
  `src/boldt_embed/autoresearch_recipe.py` — versus the protected surfaces (scoring, gates, eval
  data, leakage checks, baselines, the base config),
- the fail-closed hard rules (train ≠ eval; leakage must be VERIFIED clean; dry-run numbers are
  plumbing only; 256-d retention ≥ 0.95).

Then brief me on the **v8+ frontier program** — the goal beyond the in-loop proxy is to **beat the
same-size peers** (e5-base 278M, LFM2.5 350M) on MTEB(deu) retrieval-core (GermanQuAD / GerDaLIR /
MIRACL / MLDR, nDCG@10). The proven lever is DATA, but a single mix hits a composition trade-off, so
the program is: balanced data → complementary specialists → merge → listwise-KL distill, each judged
by the frontier gate. The new levers, all user-driven (no autonomous training):

- `/ar-data <src:w,…>` — compose a leakage-clean training mixture from `configs/data_sources.json`
  (the catalogue) and make `data_mixture` REAL for the next `/ar-run real`.
- `/ar-specialist <source>` — train one domain specialist from a shared warm-start (for merging).
- `/ar-merge <ckptA,ckptB,…>` — soup/SLERP complementary checkpoints + eval + frontier gate.
- `/ar-distill <base>` — listwise-KL distillation from the Qwen3-Reranker teacher lists.
- `/ar-mteb <model>` — the promotion eval: MTEB(deu) retrieval-core + the same-size-peer frontier gate.

Current protected-surface integrity status:

!`conda run -n boldtembed python scripts/check_autoresearch_integrity.py --format json 2>/dev/null || echo '{"status":"unknown"}'`

Then tell me the single best next action. For the in-loop proxy: `/ar-status` → `/ar-trial dry` →
`/ar-tune`. For the frontier program, the cheapest high-value first move is the **merge early-test**
(`/ar-merge outputs/v8/swimir-12k/checkpoint,outputs/v8/diverse-causal/checkpoint`) — it needs only
existing checkpoints and tells us whether specialist→merge escapes the trade-off. Do not change anything.
