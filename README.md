# Boldt-Embed-DE

A German-first embedding model **family** based on [`Boldt/Boldt-DC-350M`](https://huggingface.co/Boldt/Boldt-DC-350M).

| Variant | Name | Role |
|---|---|---|
| Causal | `Boldt-Embed-DE-350M-v1-causal` | Decoder embedder, EOS/last-token pooling |
| Bidirectional | `Boldt-Embed-DE-350M-v1-bi` | LLM2Vec/MNTP-style bidirectional adaptation |
| Reranker | `Boldt-Reranker-DE-350M-v1` | German cross-encoder for reranking & distillation |

## Status (honest)

This repository is an **engineering scaffold**. The importable core, unit tests, smoke
tests, and the local toy benchmark run on the **Python standard library only** — no
GPUs, no model weights, no third-party wheels required. The training and real-benchmark
code paths are implemented as **runnable dry-runs** (config parsing, input/shape wiring,
tiny loops) and become real training once `torch`/`transformers` and licensed German
data are available.

**No benchmark number in this repo is a claim about final model quality** unless it was
produced by a saved command recorded under `outputs/` with full run metadata. The local
benchmark validates *plumbing*, not model quality.

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
python scripts/train_causal.py        --config configs/training_causal.json --dry-run
python scripts/train_bidirectional.py --config configs/training_bidirectional.json --dry-run
python scripts/train_reranker.py      --config configs/training_reranker.json --dry-run
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
