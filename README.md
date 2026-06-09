# Boldt-Embed-DE

A German-first embedding model **family** based on [`Boldt/Boldt-DC-350M`](https://huggingface.co/Boldt/Boldt-DC-350M).

| Variant | Name | Role |
|---|---|---|
| Causal | `Boldt-Embed-DE-350M-v1-causal` | Decoder embedder, EOS/last-token pooling |
| Bidirectional | `Boldt-Embed-DE-350M-v1-bi` | LLM2Vec/MNTP-style bidirectional adaptation |
| Reranker | `Boldt-Reranker-DE-350M-v1` | German cross-encoder for reranking & distillation |

## Status (honest)

Two layers:

1. **Stdlib scaffold** — the importable core, unit tests, smoke tests, and the local toy
   benchmark run on the **Python standard library only** (no GPU/weights/wheels needed).
2. **Real GPU path** — `scripts/run_real_training.py` performs an *actual* training run
   (real forward/pool/contrastive/backward on the base weights) and a real before/after
   evaluation. It was executed on an **NVIDIA RTX A6000** on 2026-05-29; see
   `outputs/real-training/real-training-report.json`.

**Verified base-model facts** (loaded on GPU 2026-05-29): `Boldt/Boldt-DC-350M` is a
**LlamaForCausalLM**, hidden_size **1024**, 24 layers, vocab 32000, max_position **2048**,
**~435M** parameters. So the 1024-d output needs no projection head, and there is no
long-context capability beyond 2048.

**Honest scale & status (read this):** this is **not** a finished, release-ready German
embedding family. What exists, concretely:
- **Causal embedder:** a real GermanQuAD run (11,494 pairs) — in-domain test nDCG@10 0.879;
  and an at-scale, **contamination-free** run (train on DT-de-dpr Wikipedia → eval on the
  **held-out, disjoint legal** GerDaLIR benchmark) via `scripts/train_disjoint_de.py`. See
  `outputs/real-training/*.json` for saved numbers + metadata.
- **Bidirectional + reranker:** real LLM2Vec/cross-encoder *pipelines* trained only at small
  scale so far — proofs, not production models. Scaling them is the next step.
- **Not done:** broad MTEB/MMTEB run, baseline comparisons, published weights, multi-dataset
  at-scale training of all three tracks. See `RELEASE_CHECKLIST.md` / `docs/audit/final-audit.md`.

Training data follows a strict **train≠eval** rule (`docs/data/training-datasets-research-2026.md`):
benchmark datasets (GermanQuAD/GerDaLIR/MMTEB) are held out; training uses non-benchmark
permissive corpora. No number is a quality claim unless produced by a saved command under
`outputs/` with run metadata; the local hashing benchmark validates *plumbing* only.

## Install

```bash
pip install -e .            # core only (stdlib) — enough for all validation gates
pip install -e ".[train]"   # + torch/transformers/peft for real training (GPU)
pip install -e ".[eval]"    # + mteb/sentence-transformers for real MTEB eval
```

## Validation gates (run on stdlib, no weights)

```bash
make validate   # python scripts/validate_repo.py --format markdown
make smoke      # python scripts/run_smoke_tests.py --format markdown
make bench      # python scripts/run_local_benchmark.py --format markdown
make test       # python -m unittest discover -s tests
make all        # everything above + write reports to outputs/
```

## Real training / evaluation (require extras + hardware + data)

```bash
# Dry-runs (no weights): validate config + wiring
python scripts/train_causal.py        --config configs/training_causal.json --dry-run
python scripts/train_bidirectional.py --config configs/training_bidirectional.json --dry-run
python scripts/train_reranker.py      --config configs/training_reranker.json --dry-run

# REAL training + before/after eval on GPU (downloads base weights)
python scripts/run_real_training.py --device-index 0 --epochs 15

# REAL public-benchmark eval of a trained model
python scripts/run_mteb_benchmark_template.py --model <path> --config benchmarks/mteb_german_tasks.json
```

## Teacher/student 2026 workflow

A distillation-based path that fixes the Wikipedia-only overfitting found in the v1 runs:
strong teachers score multi-domain German data, and the Boldt student is trained to match
them. **Teacher execution requires the `train` extras + a GPU** (the 48 GB RTX 6000 profile);
the configs and validation below run on stdlib alone.

Configs:

- `configs/teacher_models.json` — `Qwen/Qwen3-Embedding-8B` + `Qwen/Qwen3-Reranker-8B`
  teacher defaults (model, backend, dtype, max_length, batch_size, instructions). Both are
  Apache-2.0, 32k context, instruction-aware. Loaded/validated by
  `boldt_embed.config_teacher.load_teacher_models_config`.
- `configs/student_training_2026.json` — Boldt student plan: bidirectional variant,
  Matryoshka dims `[1024,768,512,256,128,64]`, loss stack (cached MNRL/GIST + Matryoshka +
  distillation + margin-MSE), `train_eval_split_policy = public_benchmarks_eval_only` (a hard
  config error if violated), single-48GB hardware profile. Loaded by
  `load_student_training_config`.

`flash-attn` is optional (`pip install flash-attn --no-build-isolation`); the teacher loader
falls back to eager attention when it is unavailable.

## Layout

```
src/boldt_embed/   # stdlib core: config, pooling, matryoshka, metrics, losses,
                   # data, hard_negatives, eval_harness, instructions, cli
                   # + lazy-torch wrappers: model_causal, model_bidirectional, reranker
configs/           # training + evaluation config templates
scripts/           # validate / smoke / bench / train-dry-run / report
data/              # schema + small toy German pairs/triples (samples only)
benchmarks/        # toy retrieval, stress cases, MTEB task list, baselines
docs/ docs/adr/    # research notes, architecture plan, ADR-001..006, data/benchmark plans
model_cards/       # Hugging Face model cards (3 variants)
tests/             # unittest suite (stdlib)
outputs/           # saved validation / smoke / benchmark reports
```

## License

Source code: Apache-2.0 (`LICENSE`). **Model weights license is separate and unresolved**
— see `docs/adr/ADR-001-base-model-and-license.md` before any release.
