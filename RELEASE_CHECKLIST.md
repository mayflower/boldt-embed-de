# Release Checklist

Gate every public release on this list (implements ADR-006). Do not publish weights until
all **blocking** items pass.

## Blocking — licensing & provenance (ADR-001, ADR-004)
- [ ] Base-weight license confirmed: `Boldt/Boldt-DC-350M` = apache-2.0 (verified 2026-05-28).
- [ ] Every training dataset has a recorded, compatible license (`data.check_licenses` clean).
- [ ] Benchmark-leakage check clean against the eval registry (`data.find_leakage`).
- [ ] True parameter count verified; published model name is accurate (350M vs 0.5B — ADR-001).
- [ ] Base `config.json` read; 1024-d output (or projection head) decision finalized (ADR-003).

## Blocking — evaluation honesty (ADR-005)
- [ ] Real MTEB run executed; results saved under `outputs/` with full run metadata
      (command, commit, model, dataset, split, metric, hardware, output_path).
- [ ] Baselines evaluated under the same harness before any comparative claim.
- [ ] Matryoshka dims reported; German stress cases reported separately.
- [ ] No tuning against public test labels; train-time validation used a private dev split.

## Blocking — artifacts
- [ ] Model card per variant complete (intended use, limitations, evaluation, license, repro).
- [ ] No overclaims (no long-context, no "best multilingual" from a 350M German model).
- [ ] SentenceTransformers-compatible export verified to load and encode.

## Non-blocking — hygiene
- [ ] `make all` green; CI green on py3.10–3.12.
- [ ] Working tree clean; release tagged.
- [ ] `AUDIT.md` completed and signed off.

## Current status (2026-05-28)
Scaffold complete and validated. **Blocking eval + licensing-of-data items are open** because
no training/benchmark run has been executed. Not releasable yet — by design.
