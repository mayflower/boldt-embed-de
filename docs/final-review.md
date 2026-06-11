# Final Project Review

Date: updated 2026-06-11.

## Checklist
| Item | Status | Evidence |
|---|---|---|
| Git status clean | ✅ | working tree clean after each milestone |
| Latest commit recorded | ✅ | see `git log` (prompt-by-prompt commits) |
| Tests passed (or failures documented) | ✅ | 238 stdlib tests + real GPU runs (`make test`) |
| Benchmark reports saved | ✅ | `outputs/baselines/real_*.json`, `outputs/real-training/*`, run cards |
| Model cards complete | ✅ | 3 variant cards + dataset card; `validate_repo` enforces sections |
| Data/license audit complete | ✅ | `docs/data/*`, `docs/audit/final-audit.md`, `validate_data_schema` |
| Release risks documented | ✅ | `RELEASE_CHECKLIST.md`, `docs/audit/final-audit.md` |
| User-facing summary accurate | ✅ | this file + `docs/benchmark-report.md` §6e–§6g |

## Executed status (2026-06)
The 2026 teacher→student workflow has been **executed and measured** — causal, bi+MNTP, and
reranker tracks all trained on the A6000 with Qwen3-8B teachers over 3,764 multi-domain
candidates, with held-out numbers + run cards (`docs/benchmark-report.md` §6e–§6g). The honest
carve-outs: **broad MMTEB is not run** and training is at **modest scale (3,764 candidates)**,
so the project is **not release-ready**.

## What is real vs pending
- **Real (measured):** causal student GermanQuAD 0.883 / DT-test 0.950 / GerDaLIR 0.0782;
  bi+MNTP beats causal in-domain (DT-test 0.967, MNTP essential); reranker lift DT-test
  0.950→0.990 but degrades GermanQuAD; Matryoshka 256-d ≈ 97% retention. `multilingual-e5-base`
  leads (0.939 / 0.994 / 0.1343).
- **Pending (blocking release):** v2 data-scale generalization (50k–250k candidates); broad
  MMTEB/GermanDPR eval; reranker generalization (anti-degradation gate); German safety/bias;
  per-source licensing/provenance; derivative-weights license.

## Recommended next actions
1. **v2-data-scale-generalization:** 50k–250k teacher-validated multi-domain candidates;
   retrain causal vs bi+MNTP; reranker on diverse candidate lists; held-out eval.
2. Broad MMTEB-de + GermanDPR/GermanQuAD eval with saved metadata (eval-only, leakage-checked).
3. German safety/bias evaluation; finalize derivative-weights license and published name.
4. Work `RELEASE_CHECKLIST.md` to green, then publish.

## Commands run (validation)
```bash
python scripts/validate_repo.py      # structure/JSON/imports/ADR+card sections -> pass
python scripts/validate_project.py   # 8-check gate -> pass
python -m unittest discover -s tests # -> OK
make all                             # gates + saved reports
```
