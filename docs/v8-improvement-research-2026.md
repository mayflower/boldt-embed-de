# v8 — closing the MTEB(deu) gap: a 2-stage rebuild (bidirectional → broad pretrain → distilled FT)

Status: **planned**. This is the experiment generation that acts on the June-2026 research review
(five parallel literature streams) and the corrected MTEB(deu) comparison. It is a **structural
rebuild**, not a knob tweak — scoped, staged, and gated.

## 1. What the measurement actually showed (corrected)

MTEB(deu) retrieval-core, nDCG@10 (`outputs/mteb/COMPARISON_de_retrieval.md`):

| Model (seq) | GermanQuAD | GerDaLIR-S | MIRACL-hn | MLDR |
|---|---|---|---|---|
| Boldt best-r7 @256 | 0.846 | 0.050 | 0.324 | 0.203 |
| Boldt best-r7 @2048 | — | **0.195** | — | **0.264** |
| e5-base (512) | 0.923 | 0.153 | 0.530 | 0.263 |
| Qwen3-0.6B (512) | 0.911 | 0.180 | 0.541 | 0.272 |
| LFM2.5-350M (512) | 0.913 | 0.150 | 0.520 | 0.237 |

**The long-doc gap was an eval-truncation artifact** (fixed for free by serving at native context —
GerDaLIR 0.050→0.195, MLDR 0.203→0.264). The **real, structural gap is short-doc general/Wikipedia
retrieval: MIRACL (0.33 vs 0.52) and GermanQuAD (0.87 vs 0.92).** Root cause across all five research
streams: Boldt is **FAQ-overfit** (one task, one domain, easy negatives), **causal + mean-pooled**
(the weakest decoder-embedder combo), and **single-stage** (no broad weakly-supervised pretrain).

## 2. The rebuild (mirrors what LFM2.5 / e5 / Qwen3 actually do)

Three compounding levers, executed as stages. Each stage gates before the next.

### Stage 0 — Bidirectional conversion (architecture)
Every same-size-or-smaller model that beats us is **bidirectional**; we are causal. Convert the
Boldt Llama decoder to bidirectional attention (LLM2Vec mask patch — **already implemented**:
`train.enable_bidirectional`, `train_modern.load_student_sentence_transformer(bidirectional=True)`,
`apply_bidirectional_to_st`), optional **MNTP** adaptation (`mask_tokens` + `data/processed/mntp_texts.jsonl`),
then retrain contrastive+distill on the existing clean data with the round-7 recipe.
- **Gotcha (correctness):** the patch is a *runtime* modification, not saved in weights — **eval must
  re-apply it** (`apply_bidirectional_to_st`) and load with **eager attention**, else a saved
  bidirectional model is silently evaluated as causal. The v8 MTEB runner adds `--bidirectional`.
- **Evidence:** LLM2Vec gains are largest on small models; isolated mask-flip ≈ +1 pt (NV-Embed),
  full recipe much larger. **Skeptic note:** for German, dedicated encoders (ModernGBERT) have beaten
  converted decoders — so this is a *general lift to validate*, not a guaranteed MIRACL fix.
  Sources: LLM2Vec (arXiv 2404.05961), NV-Embed (2405.17428), LFM2.5 (liquid.ai/blog/lfm2-5-retrievers).

  **Stage-0 attempt 1 — NEGATIVE (recorded).** Naive flip + 300 contrastive steps, **no MNTP**,
  produced a *severely undertrained* model (not collapsed: embeddings L2-normal, weakly
  discriminative — correct match ~0.15 cos, train_loss stuck ~13.4), so MTEB cratered:
  GermanQuAD 0.048 / GerDaLIR 0.003 / MIRACL 0.005 (vs causal round-7 0.866 / 0.085 / 0.326). This
  **confirms the literature warning** — the bidirectional flip resets representation quality and
  needs (a) an **MNTP adaptation stage** (`train.mask_tokens` + `data/processed/mntp_texts.jsonl`,
  from the ORIGINAL `Boldt/Boldt-DC-350M` base which still has the lm_head) and (b) **far more
  contrastive training** to rebuild. Bidirectional is therefore high-cost / uncertain-reward for
  this German 350M decoder (cf. ModernGBERT > converted decoders); the data levers (Stages 1–2) are
  lower-risk and target the actual measured gap. The undertrained checkpoint was pruned (logged
  negative; see `outputs/mteb/v8-stage0-bi/` summary).

  **Stage-0 FINAL verdict — bidirectional DROPPED (clean A/B, evidenced).** With MNTP fixed, a proper
  A/B — both from the raw `Boldt/Boldt-DC-350M`, 4000 contrastive steps, same clean data, @512:

  | | GermanQuAD | GerDaLIR | MIRACL | MLDR |
  |---|---|---|---|---|
  | causal control (4k) | **0.855** | **0.080** | 0.297 | **0.227** |
  | bidirectional MNTP+4k | 0.767 | 0.046 | **0.330** | 0.138 |

  **Causal wins 3/4** (loses only MIRACL by 0.033) *despite the bidirectional arm spending extra MNTP
  compute* — bidirectional is a **net loss** for this German 350M decoder (confirms the ModernGBERT
  caveat). Also causal-from-raw +4k (0.855) ≈ causal round-7 (0.866), and 1500→4000 steps barely
  moved GermanQuAD (0.766→0.767): the model is **data-saturated**. **Conclusion: keep the CAUSAL
  architecture; the remaining gap to competitors is entirely DATA. Stages 1-2 proceed on the causal
  model; Stage 0 (bidirectional) is closed as an evidenced dead-end.**

### Controlled data-SCALE test (run; informative result)
Before building Stage 1, a clean controlled test: same raw base / causal / 4000 steps / recipe /
hard-negs, varying **only** the pair count (22k → 69k unique, leakage-scanned: 3513 dropped incl.
3407 WebFAQ). @512:

| | GermanQuAD | GerDaLIR | MIRACL | MLDR |
|---|---|---|---|---|
| Arm A (22k) | 0.855 | 0.080 | 0.297 | 0.227 |
| Arm B (69k, 3.1×) | 0.836 | **0.151** | 0.254 | 0.226 |

**Result: scale of GENERIC local data is NOT the MIRACL lever.** 3.1× more data *helped* GerDaLIR
(+0.071, nearly 2×) but *hurt* MIRACL (−0.043) and GermanQuAD (−0.019) — the added web/QA pairs
**diluted** the short-doc relevance. Composition matters, not raw volume. Combined with Stage-0:
**architecture (bidirectional) = net loss; training budget = saturated; generic data scale = mixed
(helps legal, hurts wiki/QA).** The MIRACL gap (0.33 → 0.52) is not crackable with local levers — it
needs TARGETED data at scale (Wikipedia ad-hoc + hard negatives + the listwise-KL objective), which
local resources can't provide, OR a reranked 2nd stage. The space is mapped; see §Conclusion.

### TARGETED online data at scale — SWIM-IR (run; **POSITIVE — the data lever, confirmed**)
Fetched **400k SWIM-IR German** pairs online (`nthakur/swim-ir-monolingual:de`, synthetic Wikipedia
ad-hoc = the MIRACL distribution), leakage-scanned (**7,595 MIRACL-overlaps dropped** → 392,398
clean — un-scanned, training on them would have inflated MIRACL). Trained causal from the raw base
on v6_clean(22k)+SWIM-IR(392k), 8000 steps, @512:

| | GermanQuAD | GerDaLIR | MIRACL | MLDR |
|---|---|---|---|---|
| v6.1 baseline | 0.843 | 0.046 | 0.332 | 0.197 |
| Arm A (22k local) | 0.855 | 0.080 | 0.297 | 0.227 |
| **SWIM-IR (414k online)** | **0.888** | 0.077 | **0.380** | **0.250** |

**MIRACL 0.332 → 0.380 (+0.048 over best prior Boldt); GermanQuAD 0.888 and MLDR 0.250 are the best
Boldt numbers yet.** Every architecture / budget / generic-local-scale experiment failed to move
MIRACL; **targeted online data at scale moved it on the first try.** This confirms the data lever
empirically. Still below competitors (0.52) — but this is only 392k (capped) + 8k steps + plain
CMNRL. Next increments: full SWIM-IR, mMARCO-de (parquet route), more steps, listwise-KL. GerDaLIR
flat (SWIM-IR is wiki, not legal). **Process lesson: this should have been the FIRST experiment, not
the last — inventory + fetch the right data before tuning architecture/budget.**

### Real online data at SCALE + DIVERSITY (run; the data lever delivered)
Fetched online (bypassing the datasets>=4 script block via direct file download) + leakage-scanned:
SWIM-IR-de 392k (wiki), **mMARCO-de 253k (web, direct TSV)**, **mqa-de 406k sample of a 17.9M source
(web FAQ, direct json.gz)** → **1.07M clean diverse pairs** (vs the original ~70k local). Causal,
16k steps, @512:

| | GermanQuAD | GerDaLIR | MIRACL | MLDR | data |
|---|---|---|---|---|---|
| v6.1 baseline | 0.843 | 0.046 | 0.332 | 0.197 | 22k local |
| SWIM-IR 12k | **0.889** | 0.074 | **0.382** | 0.254 | 414k wiki |
| diverse 16k | 0.882 | **0.107** | 0.371 | 0.252 | 1.07M wiki+web+FAQ |

**Real fetched data lifted EVERY task over the v6.1 baseline (+0.05–0.06 each):** MIRACL 0.332→0.382,
GerDaLIR 0.046→0.107, GermanQuAD 0.843→0.889, MLDR 0.197→0.254. **Composition trade-off confirmed:**
pure-wiki (SWIM-IR) maximizes MIRACL/GermanQuAD; adding web+FAQ diversity maximizes GerDaLIR but
dilutes MIRACL. No fixed ~1M mix wins all tasks at once — the competitor edge (all tasks high) needs
*far* more data + the listwise-KL objective. **Still below competitors on MIRACL (0.38 vs 0.52).**
Next: uncap the sources (full SWIM-IR/mMARCO, more of the 17.9M mqa) + listwise-KL (Stage 2, untouched)
+ per-task/balanced mixing. **Arc lesson: fetching real targeted data should have been step 1 — it
moved every metric, where architecture/budget/local-recombination did not.**

### Stage 2 — listwise-KL distillation (run; modest positive, domain-shaped)
Pure listwise-KL FT (`scripts/train_listwise_kl.py`) from the diverse model over the cached
teacher-scored lists (`reranker_train_lists_teacher_scored.jsonl`, 7500, each with a precomputed
`teacher_softmax_target`). 1500 steps, top-24, lr 5e-6, grad-checkpointing. @512:

| | GermanQuAD | GerDaLIR | MIRACL | MLDR |
|---|---|---|---|---|
| diverse base | 0.882 | **0.107** | 0.371 | 0.252 |
| + listwise-KL | 0.887 | 0.086 | **0.385** | 0.255 |

**MIRACL → new v8 best 0.385; GermanQuAD/MLDR up; GerDaLIR down (0.107→0.086).** The objective lever
works but is shaped by the teacher-list domain (mostly short-doc QA german_stress/dt → sharpens
short-doc ranking, costs legal/long). Per-batch loss is very noisy (B=4 over heterogeneous lists:
0.04↔2.0) — judge by eval, not loss. **Higher-value next: score the NEW SWIM-IR/mMARCO data with the
Qwen3-Reranker teacher (a ~1M-list inference job) and distill on THAT — domain-matched + at scale.**

## v8 CONSOLIDATED (best per task across all runs)
| | GermanQuAD | GerDaLIR | MIRACL | MLDR |
|---|---|---|---|---|
| v6.1 baseline | 0.843 | 0.046 | 0.332 | 0.197 |
| **best v8** | **0.889** | **0.107** | **0.385** | **0.255** |
| competitors | 0.91-0.92 | 0.15-0.18 | 0.52-0.54 | 0.24-0.27 |
Real fetched data (SWIM-IR/mMARCO/mqa, 1.07M) + listwise-KL closed most of the GermanQuAD/MLDR gap
and ~⅓ of MIRACL; GerDaLIR/MIRACL still trail. The two open levers to close the rest: (1) MORE data
(uncap SWIM-IR/mMARCO, the 17.9M mqa) with balanced per-task mixing to avoid the dilution trade-off;
(2) teacher-scored listwise-KL on that NEW data.

### Stage 1 — Broad weakly-supervised contrastive pretrain (data breadth)
The missing stage. mE5 = ~1B pairs @ 32k batch; Qwen3 = 150M synthetic; we have **0**. Build a broad
German/multilingual pair corpus from **local non-eval assets** — `qa_passage_non_eval_union.jsonl`
(150M), the teacher shards, German Wikipedia/title-body, mC4-de title→body — and run large-batch
in-batch-negative InfoNCE (GradCache/CachedMNRL) on the **bidirectional** model from Stage 0.
- Target scale realistic for one A6000: **20–100M pairs**, effective batch **~2–4k** (Inf-CL shows
  gains saturate past that — do **not** chase 32k).
- **Hard rule:** leakage-scan every source (`run_full_leakage_scan.py`); `public_benchmarks_eval_only`.
- Sources: mE5 (2402.05672), Inf-CL (2410.17243).

### Stage 2 — Supervised FT: listwise-KL distillation + clean negatives + multi-task (signal + breadth)
Replace margin-MSE with **listwise-KL distillation** over the **existing teacher-scored candidate
lists** (`data/processed/v6/reranker_train_lists_teacher_scored.jsonl` — query + candidates already
carry reranker scores). Two 2025 papers show plain contrastive FT *degrades* broad retrieval while
listwise-KL *improves* it (the exact FAQ-overfit symptom). Plus:
- **Positive-aware hard-neg filtering** (NV-Retriever 95% rule: drop candidates scoring ≥0.95× the
  positive; 7–15 negs, not raw top-50) — guards false negatives.
- **Multi-task / multi-domain mixture**, FAQ down-sampled to **≤30%**, add general/Wikipedia QA
  (GermanDPR-style) + synthetic German queries (E5-mistral/Gecko recipe via the Qwen3 teacher).
- **τ 0.05→0.02**, instruction prefixes (asymmetric, query-side).
- Sources: listwise-KL (2505.19274, 2502.19712), NV-Retriever (2407.15831), GISTEmbed (2402.16829),
  Qwen3 report (2506.05176), Gecko (2403.20327), E5-mistral (2401.00368).

## 3. Promotion gate (`scripts/check_v8_gate.py`)
A v8 stage is "promotable" only if, measured **directly under the MTEB(deu) harness at native
context** (not a serving wrapper):
- **PRIMARY (headroom):** MIRACL-hn nDCG@10 improves materially over v6.1 (target: close ≥⅓ of the
  0.33→0.52 gap, i.e. ≥ ~0.39) **and** GermanQuAD does not regress.
- **Do-not-regress:** GerDaLIR / MLDR at native context ≥ v6.1-native − 0.005; WebFAQ recall@100
  (the existing primary) ≥ 0.97; 256-d Matryoshka retention ≥ 0.95.
- **Fail-closed:** leakage VERIFIED clean on every training source; bidirectional eval-patch applied
  (else the number is invalid). Each reported number needs a saved run card (ADR-005).

## 4. Honest ceiling
Matching models trained on far more data/compute is **not guaranteed at 350M**. Realistic aim: close
**most** of the MIRACL/GermanQuAD gap and reach same-class parity with e5-base/LFM2.5, not beat
Qwen3-0.6B everywhere. If the single-vector dense ceiling blocks MIRACL after all stages, the
structural answer is a multi-vector/late-interaction or reranked second stage (we already have the
Boldt reranker) — not more dense contrastive training. (arXiv 2508.21038.)

## 5. Sequencing & cost
Stage 0 (foundational, ~hours) → gate → Stage 1 (GPU-days, data prep) → Stage 2 (GPU-day + listwise
loss) → gate. The AutoResearch loop can still hill-climb the *cheap* Stage-2 knobs (τ, #negs, mining
margins) but the architecture/pretrain/data changes are recipe-level, outside its editable surface.
Full citations: see `outputs/` research notes and the inline arXiv IDs above.

## 6. Driving the program — the instrumented harness (no autonomous training)
The findings above are now operationalized as user-driven slash commands (`AUTORESEARCH.md` §"The
v8+ frontier program"). Nothing trains on its own — you trigger each GPU step. The map:

| Phase | Command(s) | What it tests / produces |
|---|---|---|
| **0 — merge early-test** | `/ar-merge outputs/v8/swimir-12k/checkpoint,outputs/v8/diverse-causal/checkpoint` | Does soup of the existing wiki-specialist (MIRACL) + diverse-specialist (GerDaLIR) keep BOTH strong tasks? Needs only on-disk checkpoints — the cheapest signal on whether specialist→merge escapes the §"composition trade-off". |
| **1 — balanced data** | `/ar-data swim_ir_de_full:0.4,mmarco_de:0.3,mqa_de:0.3` → `/ar-run 1 real` → `/ar-mteb` | Does an uncapped, domain-balanced, **materialized** mixture (the `_materialize_data_mixture` hook, fail-closed on unscanned sources) beat the capped diverse-16k mix on the aggregate? |
| **2 — specialists + merge** | `/ar-specialist <src>` ×N from a shared warm-start → `/ar-merge` → `/ar-mteb` | Train one expert per domain (all warm-started from `diverse-causal` so they share a basin), then merge — the structural escape from the single-mix trade-off. |
| **3 — distill** | `/ar-distill <merged> existing` (cheap) then `new:<src>` (teacher-scored, expensive) → `/ar-mteb` | Listwise-KL sharpens ranking; §Stage-2 showed it is **domain-shaped** (lifts the teacher-list domain, can cost others) — the frontier gate catches the regression. |

The promotion bar is **`scripts/check_mteb_frontier_gate.py`** (a protected `check_*`, run post-eval
outside the loop): a candidate is `promotable` only if its 4-task aggregate ≥ the same-size-peer
frontier (max of e5-base / LFM2.5) **and** no per-task regression below the @512 baseline − tol
**and** leakage clean. Every source the harness may train on is enumerated in
`configs/data_sources.json` with its `leakage`/`training_usable` flags; the recipe is fail-closed on
anything not `scanned_clean`. This keeps the §1–§4 discipline (train≠eval, no committed weights, no
benchmark claim without a saved `outputs/mteb/<label>/summary.json`) intact under interactive driving.
