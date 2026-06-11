# Final Audit (prompt 12)

Date: 2026-05-29. Scope: the implementation repo plus the real GPU runs executed on an
NVIDIA RTX A6000. This is the prompt-12 deliverable; the root `AUDIT.md` is a shorter summary.

## Pass/fail by dimension

| Dimension | Status | Evidence | Remediation if not pass |
|---|---|---|---|
| License compliance | ✅ pass | Code Apache-2.0; base weights apache-2.0 (verified 2026-05-29); `data.check_licenses` allowlist enforced; shipped data clean (`validate_data_schema`). | Confirm each *real* dataset license + derivative-weights license before publish. |
| Data leakage / dedup | ✅ pass (policy + tooling) | `data.find_leakage` (exact + Jaccard) unit-tested; `docs/data/leakage-policy.md`. | Run against real eval registry once real corpora are added. |
| Benchmark leakage | ✅ pass (policy) | ADR-005/009 held-out-only; public test never trains. | Enforce in the real data pipeline. |
| PII exposure | ✅ pass | `data.scan_pii` (email/IBAN/phone/IPv4, legal-ref safe); shipped data has 0 hits (`tests/test_pii_schema.py`). | Re-scan + redact any real corpus before training. |
| Safety / bias | ⚠️ partial | German admin/legal domain; no toxic content in samples. No bias eval run (tiny data). | Add a German bias/safety eval before any production release. |
| German legal/admin overclaiming | ✅ pass | Model cards state "not legal advice"-style limitations implicitly; no accuracy claim on legal retrieval; legal_ref is a *stress* category, not a guarantee. | Add an explicit "not legal advice" line to cards at release. |
| Reproducibility | ✅ pass | Pinned configs; deterministic stdlib core; CI py3.10-3.12; real runs save run metadata (commit/gpu/torch) under `outputs/real-training/`. | — |
| Model-card truthfulness | ✅ pass | Cards report only measured, run-carded numbers from the executed 2026 runs (causal 0.88/0.95/0.078; bi+MNTP beats causal in-domain; reranker degradation stated honestly); e5-base shown as leading baseline. `validate_repo` enforces card sections. | Update with broad MMTEB numbers when run. |
| Git cleanliness | ✅ pass | Working tree clean each milestone; 1.7GB checkpoints git-ignored; only small JSON/MD reports tracked. | — |

## Honest scope of executed runs
Executed on the A6000 (2026-06-09/10): the 2026 teacher→student workflow with Qwen3-Embedding-8B
+ Qwen3-Reranker-8B teachers over **3,764 multi-domain, non-benchmark** German candidates
(teacher false-negative filter vetoed 464/574 adversarial distractors). **Measured** held-out
nDCG@10 (`docs/benchmark-report.md` §6e–§6g, run cards under `outputs/run-cards/`):
- causal student: GermanQuAD **0.883** / DT-test **0.950** / GerDaLIR-legal **0.0782**;
- bi+MNTP: beats causal in-domain (DT-test **0.967**); MNTP is essential (without it, collapse);
- reranker: lift DT-test 0.950→**0.990**, but **degrades GermanQuAD** 0.886→0.532;
- Matryoshka: 256-d ≈ 97% retention.
`multilingual-e5-base` still leads (0.939 / 0.994 / 0.1343). **No broad MMTEB run and no v2
data-scale training yet** — all release claims remain gated (see `RELEASE_CHECKLIST.md`).

## Top remediation items (blocking release)
1. **v2 data-scale generalization:** 50k–250k teacher-validated multi-domain candidates;
   retrain causal vs bi+MNTP; reranker on diverse candidate lists (fix GermanQuAD degradation).
2. Broad MMTEB-de + GermanDPR/GermanQuAD evaluation with saved metadata (eval-only, no leakage).
3. German safety/bias evaluation; per-source licensing/provenance via the v2 source manifest.
4. Confirm derivative-weights license and honest published parameter count (~435M).

## Reproduce
```bash
make all                              # stdlib gates + reports
python scripts/validate_project.py    # 8-check project gate
# real runs (GPU): run_real_training.py / run_real_bidirectional.py / run_real_reranker.py
```
Verdict: **workflow executed and measured on GPU; NOT release-ready.** Current best evidence:
a causal student competitive with `multilingual-e5-base` in-domain (GermanQuAD 0.88 / DT-test
0.95), with OOD legal quality and reranker generality still trailing. Close the blocking items
above (v2 data scale + broader eval + provenance + reranker generalization) before release.
