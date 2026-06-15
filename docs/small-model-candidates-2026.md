# Small-model candidates (2026): Boldt vs Qwen3-0.6B

Answers one question for v5: **is Boldt actually the best *small* base for this German RAG job, or
is `Qwen/Qwen3-Embedding-0.6B` / `Qwen/Qwen3-Reranker-0.6B` better?** Qwen3-0.6B is also small, but
multilingual, instruction-aware, 32k-context, and already trained for embedding/reranking — so
Boldt has to **earn** the production default, not be assumed.

- Config: `configs/v5_small_model_candidates.json`
- Core (stdlib): `src/boldt_embed/small_model_candidates.py` (validation, storage/params, gate)
- ML measurement + LoRA (lazy): `src/boldt_embed/small_model_measure.py`
- Orchestrator CLI: `scripts/eval_small_model_candidates.py`

## Candidates (same harness)

- **Dense**: `boldt-causal-v3` (boldt), `qwen3-emb-0.6b` (qwen3), `bge-m3` (bge),
  `e5-base` (e5).
- **Reranker**: `boldt-reranker-v4` (boldt), `qwen3-rerank-0.6b` (qwen3).

The config validator **requires ≥ 2 model families** per list — you cannot run a "comparison" with
one family.

## Modes (compose)

| flag | effect |
|---|---|
| *(default)* | baseline-only evaluation of every candidate |
| `--tune-reranker` | LoRA-tune `Qwen3-Reranker-0.6B` on v5 teacher-scored lists, then eval it |
| `--tune-embedding` | LoRA-tune `Qwen3-Embedding-0.6B`, then eval it |
| `--full-finetune` | allow full fine-tune instead of LoRA (otherwise **LoRA only**) |

Tuning teacher = **Qwen3-8B** embedding/reranker; training lists come from the v5 teacher-scored
candidates. A tuned model is added as a new candidate (`family: qwen3-tuned`) and measured under
the **same** harness as everything else.

## What gets measured (real run)

For every candidate, under one harness:

- **quality** — dense: nDCG@10 as a retriever over an eval set + corpus; reranker: nDCG@10 over
  the **fixed** candidate lists;
- **latency** (ms/query), **throughput** (qps);
- **VRAM** (peak MB), **params** (M);
- **storage** bytes/vector at each embedding dim (fp32/fp16), and dense **256-d retention**
  (nDCG@10 at 256-d ÷ full-dim) for the Matryoshka small-vector story.

`--dry-run` imports **no ML** and emits the plan (candidates, modes, gate, storage table) so you can
confirm wiring before spending GPU.

## Selection gate (family-blind)

`select_default()` picks the production default by **quality, then latency** — never by family:

1. require **≥ 2 candidates measured under the SAME harness** (else `insufficient_comparison` /
   `inconsistent_harness` — *no model is promoted without a same-harness comparison*);
2. exclude candidates over the latency budget (dense vs reranker budgets differ) or, for dense,
   below `min_256d_retention` (0.95);
3. among the survivors, take the highest quality; if a faster candidate is within
   `tie_break_quality_delta` (0.005 nDCG@10) of the best, prefer the **faster** one.

The chosen model's family is reported for transparency but is **not** an input to the decision —
unit-tested by swapping family labels and confirming the pick is unchanged.

## CLI

```
# plan only (no ML)
python scripts/eval_small_model_candidates.py --dry-run --tune-reranker

# real baseline bake-off (dense + reranker), then gate
python scripts/eval_small_model_candidates.py \
  --dense-eval data/eval/v5/dense_eval.jsonl --dense-corpus data/eval/v5/corpus.jsonl \
  --reranker-eval data/processed/v5/webfaq_eval_lists.jsonl \
  --report outputs/v5-small-rag/small_model_candidates_report.json

# add LoRA-tuned Qwen3-0.6B reranker to the bake-off
python scripts/eval_small_model_candidates.py --tune-reranker \
  --reranker-eval data/processed/v5/webfaq_eval_lists.jsonl \
  --report outputs/v5-small-rag/small_model_candidates_report.json
```

The real run is GPU-bound; per-candidate failures (e.g. a backend that needs refinement on first
run) are captured in `report.errors` rather than aborting the whole bake-off.

## Acceptance

- v5 can answer "is Boldt actually better than Qwen3-0.6B for this RAG job?" — same-harness quality
  + latency + VRAM + params + throughput + storage for every candidate.
- No model is promoted without a same-harness comparison, and the default is chosen by
  quality/latency, not model family.
