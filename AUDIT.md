# Final Audit — Boldt-Embed-DE

Date: 2026-05-28. Scope: the scaffold built in this repository (no training/benchmark run
has been executed — see "Honest scope" below). This audit red-teams licensing, leakage,
validation honesty, reproducibility, and overclaims.

## Honest scope
This environment has **no GPU, no model weights, and no licensed German training corpora**,
so the deliverable is a **validated engineering scaffold**: runnable, importable, tested
code for all three tracks; deterministic stdlib validation gates; config dry-runs; an eval
harness; and release artifacts. **No model was trained and no real benchmark was run.** Every
claim below is scoped to that reality.

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
1. No real training, weights, or MTEB numbers (out of scope here). Highest-impact follow-up.
2. Base architecture internals unverified (hidden size, vocab, context, param count). ADR-003.
3. Bidirectional attention enablement is best-effort; production should use `llm2vec`.
4. `HashingEncoder` is a deterministic stand-in, not a semantic model.
5. Real German corpora and their licenses are not yet selected (DATA_PLAN is policy + toy data).

## 7. Reproduce this audit
```bash
make all            # validate + smoke + bench + test + write reports
cat outputs/SUMMARY.md
```
Verdict: **scaffold PASS; NOT releasable as a model** until the OPEN/blocking items in
`RELEASE_CHECKLIST.md` are resolved with a real training + evaluation run.
