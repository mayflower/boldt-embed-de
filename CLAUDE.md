# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is the **implementation** repository for the German-first embedding model family
based on `Boldt/Boldt-DC-350M`. It was bootstrapped from the Boldt prompt pack.

## Mission

Ship three auditable, reproducible artifacts:

1. `Boldt-Embed-DE-350M-v1-causal`
2. `Boldt-Embed-DE-350M-v1-bi`
3. `Boldt-Reranker-DE-350M-v1`

…together with research notes, ADRs, a data/license/leakage plan, training code,
an evaluation harness, validation gates, model cards, and a final audit.

## Non-negotiable rules

- The importable core and all validation gates MUST run on the Python standard library
  only. `torch`/`transformers` are optional extras, imported lazily.
- Inspect before editing. Keep a task list. Small, reviewable commits per milestone.
- **Never claim a benchmark result unless the command was run and its output saved under
  `outputs/` with run metadata.** Separate toy/local plumbing from real model benchmarks.
- Do not commit model weights, large datasets, or secrets.
- Keep licensing and benchmark-leakage concerns visible (ADR-001, ADR-004, ADR-005).
- German-first design: query/task instructions, Matryoshka dims, and German hard
  negatives (compounds, negation, legal refs, dates/numbers, regional variants, entities).

## Commands

```bash
# Validation gates (stdlib only — no GPU/weights/wheels). Run before claiming a milestone.
make validate          # scripts/validate_repo.py        — structure / JSON / imports
make smoke             # scripts/run_smoke_tests.py       — deterministic CPU checks
make bench             # scripts/run_local_benchmark.py   — toy German retrieval (plumbing only)
make test              # python -m unittest discover -s tests
make all               # validate + smoke + bench + test + write reports to outputs/
make validate-release  # scripts/validate_release_2026.py — provenance/overclaim/weights release gate

# Run a single test module / case
python -m unittest tests.test_pooling
python -m unittest tests.test_pooling.TestMeanPooling.test_masking

# Lint (dev extra: pip install -e ".[dev]")
ruff check src scripts tests

# Config dry-runs (stdlib — parse config + wire instruction inputs, no weights)
make dry-run-causal    # or: python scripts/train_causal.py --config configs/training_causal.json --dry-run
make dry-run-bi
make dry-run-reranker

# Real training / eval (needs extras + GPU + data; downloads base weights)
pip install -e ".[train]"   # torch/transformers/peft/accelerate/datasets — for training
pip install -e ".[eval]"    # mteb/sentence-transformers — for real MTEB eval
python scripts/run_real_training.py --device-index 0 --epochs 15
python scripts/run_mteb_benchmark_template.py --model <path> --config benchmarks/mteb_german_tasks.json
```

Most `scripts/*.py` accept `--format markdown|json` and a `--dry-run` where applicable.
`flash-attn` is optional; the teacher loader falls back to eager attention when absent.

## Architecture

**Two-layer design (the single most important thing to understand):**

1. **Stdlib core** under `src/boldt_embed/` — config, pooling, matryoshka, metrics, losses,
   data, hard-negative mining, eval harness, BM25, leakage index, all the gate logic. Depends
   on the Python standard library *only*, so every validation gate, dry-run, and unit test runs
   with no GPU, no weights, and no third-party wheels. **Do not add a torch/transformers import
   at module top level here** — it breaks the gates and CI.
2. **Lazy-torch wrappers** — `model_causal.py`, `model_bidirectional.py`, `reranker.py`,
   `train*.py`, `teacher.py`, `llm2vec_boldt.py`, `reranker_modern.py`. These import
   `torch`/`transformers` *inside functions* (`_load`/`encode`/`fit`) and raise a clear
   "install the `[train]` extra" error if missing. Each exposes a stdlib `dry_run()` path that
   validates config + input wiring without loading weights (see `CausalEmbedder.dry_run`).

CI (`.github/workflows/ci.yml`) and `make all` run on stdlib alone; real training was executed
out-of-band on an RTX A6000 and its results committed as JSON under `outputs/`.

**Config system** (`config.py`): dataclasses + pure-stdlib validators for the three tracks
(causal / bidirectional / reranker) and evaluation. `validate_config_dict` dispatches by
`variant` (or by the presence of a `metrics` key for eval configs) and returns a problem list
without raising; `load_*_config` raises `ValueError` on invalid input. Config JSON lives in
`configs/`; per-experiment configs in `configs/experiments/`. Newer experiment tracks add their
own config modules (`v6_1_dense_config.py`, `v5_rag_config.py`, etc.).

**Experiment versioning (v1 → v6) is the spine of this repo.** Work proceeds in numbered
experiment generations, each with: a plan doc (`docs/v<N>-*.md`), an `configs/experiments/`
config, training/eval scripts (`scripts/*v<N>*.py`), a results doc, and a **promotion gate**.
Read `README.md` "Status (honest)" and `docs/experiment-registry.md` first — they record what
each generation concluded. Current scope (v6): a dense German RAG embedder + a standalone
reranker, each measured **directly under the harness**, not via a serving wrapper.

**Provenance & promotion gates (why numbers here are trustworthy):**

- **Run cards** (`experiment_registry.py`): every real training/eval/teacher-cache run emits a
  JSON provenance record to `outputs/run-cards/<run_id>.json` (command, commit, hardware, lib
  versions, inputs, outputs, metrics). Dry-runs emit nothing. A metric without a run card is not
  a claim.
- **Promotion gates** (`scripts/check_*_gate.py`, `check_*_promotion_gate.py`): a model variant
  may only be called "recommended"/"promoted" once its gate passes. Gates encode hardness-aware
  thresholds — improvements are judged on sets with real headroom (e.g. WebFAQ held-out), while
  **near-ceiling sets (GermanQuAD/DT-test, oracle ≥0.98) carry only a do-not-regress tolerance
  and are never the primary promotion signal** (reranking near-perfect lists only churns them).
- **`validate_release_2026.py`** is the ship gate: refuses committed weights/teacher-caches,
  enforces model-card provenance/limitation sections, bans overclaim phrases, and (critically)
  blocks any "recommended" claim that leans on a **policy-gated / bounded / rerank-or-abstain
  serving wrapper** — those are DIAGNOSTIC ONLY and never promotion evidence.

**Data discipline:** strict **train ≠ eval**. Benchmark datasets (GermanQuAD / GerDaLIR /
MMTEB) are held out; training uses non-benchmark permissive corpora. The
`public_benchmarks_eval_only` split policy is a hard config error if violated. Leakage is
checked by `leakage_index.py` / `scripts/run_full_leakage_scan.py`. See `docs/data/`.

**Teacher → student workflow:** strong teachers (`Qwen3-Embedding-8B` + `Qwen3-Reranker-8B`,
both Apache-2.0, configured in `configs/teacher_models.json`) score multi-domain German data;
the Boldt student is trained to match them (`config_teacher.py`, `teacher.py`,
`teacher_calibration.py`, cached scores under `outputs/teacher-cache/`, never committed).

**Verified base-model facts** (`Boldt/Boldt-DC-350M`, loaded on GPU 2026-05-29):
`LlamaForCausalLM`, hidden_size **1024** (so 1024-d output needs no projection head), 24 layers,
vocab 32000, max_position **2048** (no long-context beyond 2048), ~435M params.

**Key directories:** `docs/adr/` (ADR-001..009 — base model/license, causal vs bi, pooling,
data licensing, benchmark protocol, matryoshka, reranker arch, train/eval split); `model_cards/`
(the three HF cards + dataset card, provenance-gated); `benchmarks/` (toy retrieval, stress
cases, MTEB task list, baselines); `outputs/` (saved reports, run cards, experiment summaries).

## Validation (run before claiming a milestone is done)

```bash
python scripts/validate_repo.py --format markdown
python scripts/run_smoke_tests.py --format markdown
python scripts/run_local_benchmark.py --format markdown
python -m unittest discover -s tests
```

## Progress report format

Files changed · Commands run · Validation · Benchmark · Latest commit · Working tree · Risks.
