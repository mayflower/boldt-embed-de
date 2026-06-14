# v3 — Real-Domain-Generalization Plan

**Status:** planning only. **No new training has been run for v3.** This document encodes the
v2 lessons as explicit gates and lays out the v3 track. Config:
`configs/experiments/v3_real_domain_generalization.json` (validated by
`boldt_embed.v3_experiment_config`).

## 1. Why v3 exists — what v2 actually showed

v2 (committed `f4c8b029`) scaled teacher-validated data ~9k→22k and added "domain diversity"
by **generating synthetic query families (faq/admin/legal/cross-lingual) over real Wikipedia
passages**. Real held-out numbers (same harness):

| nDCG@10 | base | v1 causal | v2 causal | e5-base |
|---|---:|---:|---:|---:|
| GermanQuAD | 0.288 | 0.883 | 0.886 | 0.939 |
| DT-test | 0.223 | 0.950 | 0.944 | 0.994 |
| GerDaLIR (legal, OOD) | 0.003 | 0.078 | **0.110** | ~0.134–0.153 |

- **The one real win:** OOD legal +41% relative (0.078→0.110). In-domain stayed flat.
- **The core failure:** the teacher **rejected** the synthetic-query domains — admin 4.8%,
  faq 5.7% of pairs scored ≥2.0 (vs web 98.8%, wiki 94.6%). So the "multi-domain" set was, in
  effect, **still web+wiki**. Matryoshka 256-d retention 0.972; reranker DT-test +0.035 but
  GermanQuAD −0.040 (promotion gate failed).

**The lesson v3 must act on:** *templated queries over Wikipedia are not real domain data.*
You cannot synthesize your way into admin/FAQ/legal generalization; the teacher sees through it.

> **v3 is NOT "more prompts over Wikipedia."** v3 is **(a) real, licensed domain corpora**,
> **(b) scalable filtering/mining** that runs over the *whole* set (not a subset), and
> **(c) a reranker that does not degrade any held-out distribution.**

## 2. v2 lessons → v3 gates (enforced by the config validator)

| v2 problem | v3 gate (hard, in the config) |
|---|---|
| Synthetic admin/faq/legal queries failed teacher validation | `domain_targets` uses **`faq_real` / `admin_real` / `legal_adjacency_real_no_eval_overlap`** — real corpora only |
| Teacher cache reported `by_license {"unknown": 44336}` | `train_only_if_license_known: true` + `success_criteria.license_unknown_rows_max: 0` |
| Leakage filtered vs GermanQuAD+DT only (O(n·m) shortcut) | `train_only_if_leakage_full_scan_complete: true` |
| Hard-neg / reranker-list mining run over a ~3.5k subset | scalable mining required (§4) — no silent subset |
| Reranker degraded GermanQuAD (gate failed) | `reranker_germanquad_delta_min: 0.0` (target +0.02) **and** `reranker_dt_test_delta_min: 0.0` |
| Public benchmarks must never train | `public_benchmarks_eval_only: true` |

## 3. Real domain data (the actual work, not synthesis)

Each source must arrive with a **concrete license** (no `unknown`/`verify`/`tbd`) and pass a
**full** leakage scan against the eval corpora before any row is admitted. Candidate sources to
vet (license verification is part of the task, fail-closed in the manifest):

- **faq_real** — German FAQ/CQA with a clear license (e.g. GermanDPR-style QA, MQA-de *only if*
  the CC0 claim is re-confirmed and a script-free loader exists; both currently BLOCKED in the
  v2 manifest — unblock only after verification).
- **admin_real** — German public-administration / forms / Verwaltungsportal text under an open
  license (e.g. GovData / Amtsblatt-style open-data corpora). Real documents, not wiki.
- **legal_adjacency_real_no_eval_overlap** — open German legal/【admin】text (e.g.
  Gesetze-im-Internet / EUR-Lex German) that is **disjoint from GerDaLIR** (full leakage scan,
  not the §-keyword wiki filter v2 used).
- **web / wiki_non_eval / german_stress / cross_lingual_de_en** — keep the v2 real sources
  (ger_backtrans, dt_de_dpr, swim_ir, adversarial) that the teacher already validated at 95–99%.

Targets (`domain_targets`, sum = 1.0): web .25, wiki_non_eval .20, faq_real .15, admin_real .15,
legal_adjacency_real_no_eval_overlap .15, german_stress .05, cross_lingual_de_en .05. Scale:
**≥100k candidates**, **≥50k teacher-validated positives** (vs v2's 44k→22k).

## 4. Scalable filtering & mining (fix the O(n·m) infra)

v2 surfaced that `data.find_leakage` and `negative_mining_2026.bm25_rank` rebuild work per
query → O(n·m), unusable at 100k. v3 requires:

- **Leakage:** **built** — `src/boldt_embed/leakage_index.py` + `scripts/build_leakage_index.py`
  + `scripts/run_full_leakage_scan.py`. Two-stage scan (exact/normalized hash + SimHash bands +
  MinHash-LSH blocking → exact token-shingle Jaccard only on blocked pairs) makes a **full** scan
  against all eval corpora (incl. GerDaLIR's 200MB) tractable; verify-stage comparisons are
  subquadratic (reported as `jaccard_comparisons` vs `naive_comparisons`). Writes
  `outputs/v3-real-domain/leakage/leakage_report.json` + `leakage_hits.jsonl`, with `--drop-hits`
  for a cleaned candidate file. Gate `train_only_if_leakage_full_scan_complete` is enforced at
  train time by `train_modern_embedder.py --require-leakage-report <report>` (refuses to train
  unless the report exists and is clean or cleaned).
- **Mining:** a BM25 index built **once** (not per query) + optional dense ANN, so hard-negative
  and reranker-candidate-list mining runs over the **whole** validated set — no ~3.5k subset.
- **Provenance:** the teacher-cache summary must report a real license histogram with
  **zero `unknown`** (`license_unknown_rows_max: 0`); carry `source`/`license` end-to-end.

## 5. Models, eval, success criteria

- Students: causal (production default per v1/v2) and bi+MNTP (comparison). Same teacher→student
  distillation + Matryoshka stack.
- Reranker: mixed-loss over distribution-aware candidate lists, but trained on **teacher-validated
  real positives** (v2's lists inherited weak synthetic positives: pos median 5.0 ≤ neg 5.7).
- Held-out eval (eval-only, never trained): GermanQuAD, DT-test, GerDaLIR — same harness as v1/v2.
- Success criteria (`success_criteria`): GermanQuAD ≥ 0.886 (hold v2), **DT-test ≥ 0.95**
  (recover v1), **GerDaLIR ≥ 0.12** (stretch 0.125 — beyond v2's 0.110), reranker deltas ≥ 0 on
  **both** held-out sets (GermanQuAD target +0.02), Matryoshka 256-d retention ≥ 0.95,
  **license_unknown_rows_max = 0**.

## 6. Sequenced steps (each gated; nothing trains until gates pass)

1. Source + license-verify real faq/admin/legal corpora → extend the manifest (fail-closed).
2. Build scalable leakage + mining infra; run a **full** leakage scan against all eval corpora.
3. Build the ≥100k candidate set (real domains, license known, leakage-clean).
4. Teacher-score the full set; confirm `by_license` has **zero unknown**; keep ≥50k validated.
5. Train causal (+ bi+MNTP); eval on the three held-out sets.
6. Train reranker on real-positive candidate lists; lift eval; **promotion gate must pass**.
7. `summarize` + release gate `--require-v2-artifacts` (reuse) → only then is v3 release-like.

## 7. Out of scope / explicit non-goals

- No more synthetic queries over Wikipedia to fill admin/faq/legal — v2 proved the teacher
  rejects them.
- No training on any source with an uncertain license, and no partial/subset leakage scan —
  both are blocked by the config gates above.
