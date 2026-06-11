# Audit summary — Boldt-Embed-DE

> **Canonical, up-to-date audit: [`docs/audit/final-audit.md`](docs/audit/final-audit.md).**
> This root file is a short summary; on any conflict the canonical file wins.

**Status (updated 2026-06-11):** the 2026 teacher→student workflow has been **executed** on
the A6000 (Qwen3-8B teachers; 3,764 multi-domain non-benchmark candidates). **Measured**
held-out nDCG@10: causal student GermanQuAD **0.883** / DT-test **0.950** / GerDaLIR-legal
**0.0782** (`multilingual-e5-base` leads at 0.939 / 0.994 / 0.1343). bi+MNTP executed (beats
causal in-domain, DT-test 0.967; MNTP essential). Reranker lifts DT-test 0.950→0.990 but
degrades GermanQuAD 0.886→0.532 (in-distribution only). Matryoshka 256-d ≈ 97% retention.
Broad MTEB/MMTEB, published weights, and v2 data-scale generalization are **not done**. **Not
release-ready** — see `RELEASE_CHECKLIST.md`, `docs/benchmark-report.md` §6e–§6g.

Date: 2026-05-28 (updated 2026-05-29). This audit red-teams licensing, leakage, validation
honesty, reproducibility, and overclaims.

## Honest scope
**Correction (2026-05-29):** an earlier version of this audit and the project's scoping
question stated "no GPU available." That was **wrong and unverified** — this box has a
Tesla P40 (24GB) and an **NVIDIA RTX A6000 (48GB)**. `nvidia-smi` was not run before scoping,
which is a process failure now fixed.

The deliverable has two layers: (1) a validated **stdlib scaffold** (runnable, tested code for
all three tracks; deterministic gates; eval harness; release artifacts) and (2) **executed
real GPU runs** of the 2026 teacher→student workflow.

**Honest scale of the executed runs (2026-06-09/10):** Qwen3-Embedding-8B + Qwen3-Reranker-8B
teachers scored 3,764 multi-domain, non-benchmark German candidates; the teacher
false-negative filter vetoed 464/574 adversarial distractors. A causal student (CachedMNRL +
Matryoshka) and a bi+MNTP student were trained and **measured on held-out** GermanQuAD/DT-test/
GerDaLIR; a reranker was trained and measured as lift over fixed candidate sets. These are
real, saved, run-carded numbers (`docs/benchmark-report.md` §6e–§6g) — **competitive with
`multilingual-e5-base` in-domain but not release-ready**: OOD legal quality and reranker
generality still trail, and there is no broad MMTEB run or v2 data-scale training yet.

**Next improvement target — `v2-data-scale-generalization`:** 50k–250k teacher-validated
multi-domain candidates, causal vs bi+MNTP retrain, reranker trained on diverse candidate
lists, held-out eval (`docs/v2-generalization-plan.md` when added).

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
- Model cards report **only measured, saved** numbers from the executed 2026 runs (with run
  cards); no SOTA/quality overclaim; `multilingual-e5-base` shown as the leading baseline. ✅
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
1. The executed run is at **modest scale** (3,764 candidates): real, measured held-out numbers
   (competitive with e5-base in-domain) but OOD legal quality and reranker generality trail.
   Highest-impact follow-up = **v2 data-scale generalization** (50k–250k candidates) + broad MMTEB.
2. Base architecture **now verified** (LlamaForCausalLM, hidden 1024, 24 layers, vocab 32000,
   ctx 2048, ~435M params) — resolves the prior ADR-003 MUST-VERIFY. 1024-d needs no projection.
3. All three tracks are **executed and measured** (causal, bi+MNTP, reranker). Open: bi+MNTP
   vs causal is decided on current evidence (causal has a slight OOD edge); the reranker still
   **degrades GermanQuAD** (in-distribution only) — a v2 anti-degradation gate is planned.
4. `HashingEncoder` is a deterministic stand-in, not a semantic model.
5. v2 training corpora + per-source licenses still to be expanded/verified (source manifest is
   the v2 mechanism); current run used DT-de-dpr / ger-backtrans-paraphrase / swim-ir / synthetic.
6. Process failure (now fixed): hardware was not probed before scoping. Lesson: run
   `nvidia-smi` / check the env before asserting constraints.

## 7. Reproduce this audit
```bash
make all            # validate + smoke + bench + test + write reports
cat outputs/SUMMARY.md
```
Verdict: **workflow executed and measured; NOT release-ready.** Current best evidence is a
causal student competitive with `multilingual-e5-base` in-domain; OOD legal and reranker
generality trail. Close the OPEN/blocking items in `RELEASE_CHECKLIST.md` (v2 data scale,
broader eval, licensing/provenance, reranker generalization) before any release.
