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
| Model-card truthfulness | ✅ pass | Cards report only saved runs; causal card shows the real tiny-run table; bi/reranker eval honestly marked tiny/no-lift. `validate_repo` enforces card sections. | Update with real MMTEB numbers when run. |
| Git cleanliness | ✅ pass | Working tree clean each milestone; 1.7GB checkpoints git-ignored; only small JSON/MD reports tracked. | — |

## Honest scope of executed runs
Real, on the A6000 (2026-05-29): causal embedder (toy ndcg@10 0.774→0.94), bidirectional
LLM2Vec (MNTP 9.31→5.45, contrastive→~0, bidirectional attention verified), reranker
(train pairwise-acc 1.0; **no lift** on unseen queries — tiny-run generalization gap).
**No production-scale training and no real MMTEB run.** All quality-bearing claims are gated
on those (see `RELEASE_CHECKLIST.md`).

## Top remediation items (blocking release)
1. Acquire + license real German corpora; run leakage/PII at scale.
2. Real MMTEB German + GermanDPR/GermanQuAD evaluation with saved metadata.
3. German safety/bias evaluation.
4. Confirm derivative-weights license and honest published parameter count (~435M).

## Reproduce
```bash
make all                              # stdlib gates + reports
python scripts/validate_project.py    # 8-check project gate
# real runs (GPU): run_real_training.py / run_real_bidirectional.py / run_real_reranker.py
```
Verdict: **scaffold + pipeline PASS and verified on GPU; NOT releasable as a model** until
the blocking items above are closed.
