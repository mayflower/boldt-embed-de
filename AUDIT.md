# Audit summary — Boldt-Embed-DE

> **Canonical, up-to-date audit: [`docs/audit/final-audit.md`](docs/audit/final-audit.md).**
> This root file is a short summary; on any conflict the canonical file wins.

**Status (2026-05-29):** scaffold + pipelines verified on GPU, with a real causal embedder
run on GermanQuAD (in-domain nDCG@10 0.879) and an at-scale contamination-free run
(DT-de-dpr → held-out **legal** GerDaLIR). The **bidirectional and reranker** tracks have
real pipelines but are trained only at **small scale** so far. Broad MTEB/MMTEB, baseline
comparisons, and published weights are **not done**. **Not releasable as a model** — see
`RELEASE_CHECKLIST.md`.

Date: 2026-05-28 (updated 2026-05-29). This audit red-teams licensing, leakage, validation
honesty, reproducibility, and overclaims.

## Honest scope
**Correction (2026-05-29):** an earlier version of this audit and the project's scoping
question stated "no GPU available." That was **wrong and unverified** — this box has a
Tesla P40 (24GB) and an **NVIDIA RTX A6000 (48GB)**. `nvidia-smi` was not run before scoping,
which is a process failure now fixed.

The deliverable has two layers: (1) a validated **stdlib scaffold** (runnable, tested code
for all three tracks; deterministic gates; eval harness; release artifacts) and (2) a **real
GPU run**: `scripts/run_real_training.py` loaded the base weights on the A6000, trained the
causal embedder (real forward/pool/InfoNCE/backward), saved a checkpoint, and evaluated
base-vs-trained on the toy benchmark (`outputs/real-training/real-training-report.json`).

**Honest scale of the real run:** 7 toy German triples, 15 epochs. The 435M model trivially
separates them (training loss → 0), so this demonstrates the pipeline and a real before/after
improvement (toy ndcg@10 0.774 → 0.94 on 8 held-out queries) — it is **not** a production
model and **not** a public-benchmark claim. Real corpora + a real MTEB run remain outstanding.

## 1. Licensing (ADR-001, ADR-004)
- **Code:** Apache-2.0 (`LICENSE`). ✅
- **Base weights:** `Boldt/Boldt-DC-350M` = `apache-2.0`, verified from the HF model card on
  2026-05-28. ✅
- **Training data:** `data.check_licenses` enforces a permissive allowlist; shipped toy
  samples pass; synthetic generation is versioned (`data/synthetic/prompt_specs.json`). ✅
- **OPEN (blocking for release):** per-dataset licenses for any *real* corpus added later;
  final derivative-weights license; true parameter count / model name (350M vs 0.5B). ⚠️

## 2. Benchmark leakage (ADR-005)
- `data.find_leakage` detects exact + near-duplicate (Jaccard) overlap; unit-tested. ✅
- Policy: public **test** splits never enter training; private dev split for train-time eval. ✅
- **OPEN:** run leakage against the real eval registry (GermanDPR/GermanQuAD, MMTEB) once a
  real training set exists. ⚠️

## 3. Validation honesty
- All stdlib gates pass: `validate_repo`, `run_smoke_tests`, `run_local_benchmark`, `unittest`
  (84 tests). See `outputs/SUMMARY.md`. ✅
- The local benchmark is labeled **plumbing-only** in code, report, README, and model cards;
  it is not presented as model quality. ✅
- Model cards state **"untrained scaffold"** and report **no** evaluation numbers; the
  Evaluation sections say *Pending*. ✅
- MTEB script writes required run metadata (command, commit, model, dataset, split, metric,
  hardware, output_path) before any result is reported. ✅

## 4. Reproducibility
- Pure-stdlib core → gates run identically without ML deps or network. ✅
- Configs pinned; deterministic generators/encoders (no RNG); CI on py3.10–3.12. ✅
- Dry-run trainers reproduce config + wiring without weights. ✅

## 5. Overclaim red-team
- Searched README / model cards / docs for quality, SOTA, long-context, and multilingual
  claims: **none made.** Non-goals (ADR-005) are stated in `ARCHITECTURE_PLAN.md` and cards. ✅
- 1024-d output is explicitly flagged **MUST-VERIFY** against the base hidden size. ✅

## 6. Known gaps / risks
1. The real run is **tiny** (7 triples): proves the pipeline, not model quality. A real run
   needs licensed German corpora at scale + a real MTEB evaluation. Highest-impact follow-up.
2. Base architecture **now verified** (LlamaForCausalLM, hidden 1024, 24 layers, vocab 32000,
   ctx 2048, ~435M params) — resolves the prior ADR-003 MUST-VERIFY. 1024-d needs no projection.
3. Bidirectional + reranker tracks are **not yet trained** (only the causal track ran);
   bidirectional attention enablement is best-effort — production should use `llm2vec`.
4. `HashingEncoder` is a deterministic stand-in, not a semantic model.
5. Real German corpora and their licenses are not yet selected (DATA_PLAN is policy + toy data).
6. Process failure (now fixed): hardware was not probed before scoping. Lesson: run
   `nvidia-smi` / check the env before asserting constraints.

## 7. Reproduce this audit
```bash
make all            # validate + smoke + bench + test + write reports
cat outputs/SUMMARY.md
```
Verdict: **scaffold PASS; NOT releasable as a model** until the OPEN/blocking items in
`RELEASE_CHECKLIST.md` are resolved with a real training + evaluation run.
