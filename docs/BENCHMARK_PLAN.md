# Benchmark Plan

Implements ADR-005. Separates **plumbing** (local, stdlib, here) from **real model
evaluation** (MTEB, requires the trained model + `eval` extras).

## Local plumbing benchmark (deterministic, no weights)
- `scripts/run_local_benchmark.py` over `benchmarks/toy_de_retrieval.json`.
- BM25 baseline + deterministic HashingEncoder stand-in (NOT Boldt).
- Reports nDCG/MRR/Recall/MAP@k and Matryoshka-by-dim for the stand-in.
- Purpose: prove the metric + Matryoshka code is correct. **Not a quality claim.**

## Real model evaluation (post-training)
- `scripts/run_mteb_benchmark_template.py --model <path>` with `[eval]` extras.
- Suite (`benchmarks/mteb_german_tasks.json`): GermanQuADRetrieval, GerDaLIRSmall, STS22,
  MassiveIntentClassification, MLDR, MIRACL. **Task names drift across MTEB versions —
  verify with `mteb.get_tasks()` first.**
- Metrics: nDCG@10, MRR@10, Recall@10/100, MAP@10 (+ Spearman/accuracy/v-measure by task type).
- Matryoshka: report each dim `[1024,768,512,256,128,64]`.
- German stress tests (`benchmarks/stress_cases_de.jsonl`): compound, legal_ref, negation,
  regional, orthography, number_date — reported separately from aggregates.

## Baselines (run under the SAME harness before any claim)
`baselines.json`: deepset-mxbai-embed-de-large-v1, multilingual-e5-large-instruct, bge-m3,
jina-embeddings-v3, Qwen3-Embedding-0.6B.

## Provenance (required)
Every reported number carries: command, commit, model, dataset, split, metric, hardware,
output_path. Held-out only — no tuning against public test labels.
