# v4 German RAG reranker runbook

One command runs the whole v4 reranker pipeline:
`scripts/run_v4_rag_reranker_experiment.py`. It orchestrates the v4 scripts (it does not
re-implement them), is **safe by default**, and **directly optimizes a RAG reranker — it does NOT
require legal/admin corpora**. GerDaLIR is diagnostic-only.

```bash
python scripts/run_v4_rag_reranker_experiment.py \
  --config configs/experiments/v4_rag_reranker.json \
  --work-dir outputs/v4-rag-reranker \
  --faq-input data/raw/v3/faq_real_local.jsonl \
  --mode full --device cuda --run-id-prefix v4 \
  --i-understand-this-runs-gpu
```

## Modes

- `dry-run` (default): validate the config, write the planned `COMMANDS.md` / `STATUS.json` /
  `V4_RAG_RESULTS.{md,json}`. **No torch, no downloads, nothing executed.**
- `smoke`: run the CPU stages, skip/print GPU stages (`--allow-ml-smoke` to run them).
- `full`: execute everything — **requires `--i-understand-this-runs-gpu`**.

## Stages

1. validate v4 config (fail-fast: legal diagnostic-only, public benchmarks eval-only).
2. build WebFAQ **held-out** eval split + 3. WebFAQ **train** split (deterministic hash split).
4. build local RAG eval — only if `data/eval/rag_local/` exists (optional).
5. BM25 index + 6. BM25 search → fixed first-stage results (the real first stage).
   *Dense (v3 causal) / e5 / qwen diversity is optional: pass prebuilt per-query result files via
   `--dense-results .../dense_{set}.jsonl` (and `--e5-results` / `--qwen-results`); they are
   merged into the candidate lists. The orchestrator never fabricates a dense retriever.*
7. build fixed candidate lists (train + each eval set).
8. teacher-score the train candidate lists (Qwen3-Reranker-8B).
9. train the v4 RAG reranker (`mixed_listwise`; `--eval-query-ids` = WebFAQ held-out, a hard
   leakage guard).
10. reranker **lift** per eval set: WebFAQ held-out, GermanQuAD, DT-test, local RAG (if present),
    GerDaLIR (only with `--with-gerdalir-diagnostic`, marked `--diagnostic`).
11. **promotion gate** (GerDaLIR ignored). 12. results summary. Run cards are written by each
    real stage.

## Safety (you cannot promote a bad RAG reranker)

- `dry-run` imports no torch and downloads nothing.
- `full` requires `--i-understand-this-runs-gpu`.
- **WebFAQ leakage**: the deterministic hash split keeps train/held-out (query, answer) pairs
  disjoint; `train_rag_reranker_v4 --eval-query-ids` hard-fails if a held-out query is in training.
- **Fixed candidates**: lift reports carry `fixed_candidates`; the gate fails if any eval set is
  not a fixed candidate list.
- **No eval-label leakage**: eval candidate lists carry `label=null` (positives via qrels only).
- **Legal/admin never block**: GerDaLIR is diagnostic-only; the gate ignores it.

A **gate** stage (promotion gate) aborts by default; with `--allow-research-failures` the run
continues but the **verdict becomes `invalid_for_promotion`** (never `promotable`).

## Outputs

`outputs/v4-rag-reranker/`: `COMMANDS.md`, `STATUS.json`, `V4_RAG_RESULTS.{md,json}`, plus every
stage's artifacts (eval splits, first-stage results, candidate lists, teacher scores, checkpoint,
per-set `reranker_lift_*.json`, `rag_reranker_gate.{json,md}`).

Verdicts: `planned` (dry-run) · `smoke-ok` · `promotable` (full, gate green) ·
`invalid_for_promotion` (gate failed under `--allow-research-failures`) · `failed`.
