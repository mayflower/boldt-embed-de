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

**Honest scale:** the executed run is a *tiny* real run on 7 toy German triples — it proves
the GPU pipeline trains and improves a real model (toy-eval ndcg@10 0.774 → 0.94), it is
**not** a production model and **not** a public-benchmark claim. No number here is a quality
claim unless produced by a saved command under `outputs/` with run metadata; the local
hashing benchmark validates *plumbing* only.

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
