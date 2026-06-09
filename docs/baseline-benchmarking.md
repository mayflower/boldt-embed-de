# Baseline & teacher benchmarking

A reproducible runner that benchmarks **baselines, teachers, and Boldt student checkpoints**
through one config and one report format — so student-vs-baseline and student-vs-teacher
comparisons are apples-to-apples. Code: `configs/baseline_models.json`,
`scripts/run_baseline_benchmarks.py`.

## Configured models (not hard-coded)

`configs/baseline_models.json` lists each model with `backend`, `query_instruction`,
`document_instruction`, `max_length`, `batch_size`, `expected_dim`, `normalize`. Defaults:

- `intfloat/multilingual-e5-base`, `intfloat/multilingual-e5-large-instruct`
- `BAAI/bge-m3`, `mixedbread-ai/deepset-mxbai-embed-de-large-v1`
- `Qwen/Qwen3-Embedding-0.6B / 4B / 8B` (the teachers, as baselines too)
- the local Boldt student (`backend: local_boldt`)

Backends: `sentence_transformers`, `local_boldt`, `transformers_custom` (adapter required),
and `local_hashing` — a deterministic stdlib char-n-gram stand-in used for plumbing/tests
only (**not** a quality claim).

## Eval modes

- **`local`** — encode a local JSONL retrieval fixture (`--eval-corpus/--eval-queries/--qrels`,
  same format as the hybrid eval) and report nDCG@10 / MRR@10 / Recall@10 / Recall@100 /
  MAP@10. Per-model failures are recorded and the run continues.
- **`mteb`** — MTEB/MMTEB via the `mteb` package when installed (task list in
  `benchmarks/mteb_german_tasks.json`; see `run_mteb_benchmark_template.py`).

Every report row carries full run metadata: command, git commit, Python, platform, and
torch / transformers / sentence-transformers / mteb versions (read from package metadata
without importing them, so `--dry-run` stays ML-free).

## Run it

```bash
# Recommended first run: a small local fixture across configured models
python scripts/run_baseline_benchmarks.py \
  --models configs/baseline_models.json --mode local --task-name gerdalir_small \
  --eval-corpus data/eval/gerdalir_corpus.jsonl \
  --eval-queries data/eval/gerdalir_queries.jsonl --qrels data/eval/gerdalir_qrels.jsonl \
  --only multilingual-e5-base boldt-modern-bi \
  --output outputs/baselines/baseline_report.json

# Plan only (no downloads, no torch)
python scripts/run_baseline_benchmarks.py --dry-run
```

`--only` filters models by substring (cheap iteration). The runner writes both
`baseline_report.json` and a `baseline_report.md` table.

## Avoiding public-test overfitting

Benchmarks here are **evaluation-only**. Never feed a benchmark's test corpus into training
(see `docs/data/training-datasets-research-2026.md`). To compare **teacher vs student**, run
both through the *same* fixture and read the gap: the teacher is the ceiling, the student is
what shipped. The deterministic `local_hashing` stand-in is for verifying the harness, not for
reporting numbers.
