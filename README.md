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

**Honest scale & status (read this):** the 2026 teacher→student workflow has been **executed**
across v1→v2→v3 (Qwen3-Embedding-8B + Qwen3-Reranker-8B teachers; real, non-benchmark German
data). Measured held-out nDCG@10 (`docs/benchmark-report.md` §6e–§6j, run cards under
`outputs/run-cards/`):
- **Dense causal v3 — current best causal retriever:** GermanQuAD **0.885**, **DT-test 0.970
  (best causal to date)**, GerDaLIR (legal, OOD) **0.089**. Trained on real teacher-validated
  data including **real WebFAQ FAQ** (which validated at **70.8%** teacher acceptance vs v2
  synthetic FAQ **5.7%**). `multilingual-e5-base` (0.939 / 0.994 / 0.153) still leads, esp. OOD.
- **Reranker — still NOT promoted:** v2 reduced GermanQuAD degradation −0.354→−0.040 but it
  remains negative, so the promotion gate (`check_reranker_promotion_gate.py`) blocks it. A
  reranker may only be called "recommended" once that gate passes.
- **Bidirectional + MNTP:** executed (MNTP essential); competitive but causal keeps the edge.
- **Matryoshka:** 256-d retains ~97% of 1024-d quality on GermanQuAD.

**`v4-rag-reranker` — CLOSED (2026-06-14, RTX A6000):** distilled from Qwen3-Reranker-8B over
7,415 WebFAQ candidate lists. Lift over fixed BM25 top-20 (nDCG@10): **WebFAQ held-out +0.2907**,
GermanQuAD **−0.0711**, DT-test −0.0007. **Promotion gate FAILED** → a strong in-domain FAQ
reranker that does **not** generalize. It stays **Experimental / not promoted** (see
`outputs/v4-rag-reranker/V4_RAG_RESULTS.md`). Two lessons: v4 trained on a single style (FAQ), and
GermanQuAD/DT-test first stages were near-ceiling (recall 0.96–0.99, oracle 1.0) so reranking only
churns near-perfect lists — a tiny negative delta there is noise, not failure.

**`v5-small-rag` — EXECUTED, now historical/diagnostic (superseded by v6)** (`docs/v5-small-rag-plan.md`,
`configs/experiments/v5_small_rag.json`): a **small, deployable** German RAG retriever + reranker
trained on **diverse** question styles (FAQ, QA-passage, web non-FAQ, long-doc, German stress,
local RAG) with **hardness-aware** candidate lists. Promotion is driven by sets with real headroom
(WebFAQ held-out ≥+0.05, local RAG/hard private web-QA ≥+0.03) plus a 256-d Matryoshka retention
≥0.95 gate; near-ceiling sets (GermanQuAD/DT-test, oracle ≥0.98) carry only a −0.005
do-not-regress tolerance and are **never the primary promotion signal**. **Legal/admin is retired
as a product target — GerDaLIR stays diagnostic-only.** v1–v4 are kept historical/diagnostic.

**`v5-small-rag` reranker — EXECUTED (2026-06-15, RTX A6000):** real multi-domain run (6,500 rows
WebFAQ + DPR-train QA/long-doc/stress, leakage-filtered vs DT-test + GermanQuAD; 113,145 teacher
pairs scored by Qwen3-Reranker-8B; listwise-KL training on `Boldt/Boldt-DC-350M`, FAQ share 0.217).
Hardness-aware gate (`outputs/v5-small-rag/V5_RESULTS.md`): WebFAQ +0.1665 (primary pass), DT-test
+0.0211 (pass), **GermanQuAD −0.0285 with 16.9% catastrophic drops → FAIL**. It improves on v4
(GermanQuAD −0.0711→−0.0285, DT-test −0.0007→+0.0211) and lifts every set strongly where there is
headroom, but **the promotion gate FAILS**. The v5 reranker is **Experimental — not recommended for
production reranking.** Failure mode: over-reranking near-ceiling first-stage lists.

**v5 policy experiments — DIAGNOSTIC ONLY (not the product):** rerank-or-abstain, a conservative
reranker with a rank-preservation loss, a preservation grid (lp04/lp06/lp08), and a bounded
`margin_override` serving policy were all executed. They are useful **diagnostics** but **not the
product goal** — we do **not** recommend a policy-gated serving workaround. Two firm conclusions:
(1) **no λ makes raw always-rerank safe** on GermanQuAD (catastrophic stays 0.11–0.18), so retraining
did not change the answer; (2) the frozen bounded policy was evaluated on a held-out near-ceiling
guardrail and **FAILED its promotion gate** (WebFAQ policy Δ **+0.0245 < +0.05**). Failure analysis
(`docs/v5-policy-failure-analysis.md`) shows the WebFAQ under-lift is **mostly first-stage recall
failure**: in 234/344 failing queries the positive is **absent from the candidate list** (BM25 never
retrieved it), so **no reranker — bounded or raw — can recover it.**

**Conclusion — scope reset to v6 (the actual product):** the **v5 reranker stays Experimental — not
recommended** (raw or policy-gated). The next target is **dense first-stage recall + standalone
reranker quality measured directly under the harness**, not a serving wrapper. See
`docs/v6-dense-rag-and-reranker-plan.md`, `outputs/v5-small-rag/V5_RESULTS.md`, and
`docs/benchmark-report.md` §6o–§6p.

**`v6` — EXECUTED (2026-06-16, RTX A6000): dense retriever is the win; raw reranker FAILS its gate.**
The Boldt dense v6 retriever **materially fixes first-stage recall** over the real WebFAQ corpus:
**Recall@100 0.651 → 0.964** (missing-positive rate 0.349 → 0.036; `docs/dense-recall-gate.md`,
`outputs/v6-dense-rag/`). A standalone reranker was then trained on multi-domain teacher-scored union
lists (449,832 Qwen3-Reranker-8B pairs, no policy loss) and evaluated **RAW**: it **FAILS the
promotion gate** — WebFAQ Δ +0.036 (< +0.05) and **GermanQuAD Δ −0.086 with 21% catastrophic drops**
(it over-reranks near-ceiling guardrail lists; `outputs/v6-reranker/raw_gate.md`). Active RAG evals:
**WebFAQ / local RAG / GermanQuAD / DT-test**; **GerDaLIR/legal is diagnostic-only**. Both models stay
**not recommended** until their gates pass: the **dense embedder** needs the dense-recall gate
(`scripts/check_dense_recall_gate.py`, currently advisory-fail on top-50), the **reranker** needs the
RAW reranker gate (`scripts/check_v6_raw_reranker_gate.py`, failed). **Policy-gated/bounded/abstain
results are diagnostic-only and never promotion evidence; no serving wrapper is required for safety**
— all enforced by `scripts/validate_release_2026.py` (`--require-v6-dense-artifacts`,
`--require-v6-raw-reranker-artifacts`). See `docs/v6-raw-reranker-gate.md`.

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
