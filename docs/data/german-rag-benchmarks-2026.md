# German RAG benchmarks — survey & recommendation (checked 2026-06-17)

Which benchmarks to evaluate the German RAG stack on, beyond our current set (WebFAQ held-out +
GermanQuAD + DT-test guardrails + GerDaLIR diagnostic). Sources are fresh web checks (2025–2026);
verify task names with `mteb.get_tasks()` before a run (they drift across MTEB versions).

## Two buckets

### A. First-stage RETRIEVAL benchmarks (what our dense gate measures: nDCG@10 / Recall@k)

| benchmark | what it is | RAG relevance | priority | leakage note |
|---|---|---|---|---|
| **MIRACL (de)** | human-annotated open-domain German **Wikipedia** retrieval (726k judgments / 78k queries across 18 langs) | **high** — the standard open-domain multilingual retrieval eval | **add (high)** | German Wikipedia — **may overlap our `wiki_non_eval` training (DT-de-dpr)**; run leakage filter |
| **mMARCO (de)** | German MS MARCO passage retrieval (machine-translated), large corpus | **high** — standard large-scale passage retrieval | **add (high)** | translated MS MARCO; not in our training |
| **MLDR (de)** | Multilingual **Long-Document** Retrieval | high for **long-context RAG** | add (med) | not in our training |
| **WikipediaRetrievalMultilingual (de)** (MMTEB) | German Wikipedia QA retrieval | high — general German Wikipedia RAG | add (med) | Wikipedia overlap risk (as MIRACL) |
| **GermanDPR** (deepset; MTEB(deu)) | open-domain German QA retrieval over Wikipedia paragraphs | high | add (guardrail) | **our DT-test / `wiki_non_eval` derive from the German-DPR family — verify the eval split is disjoint** |
| **GermanQuAD-Retrieval** (MTEB(deu)) | German QA passage retrieval | medium (already used) | keep | held out |
| **XMarket (de)** (MTEB(deu)) | e-commerce product retrieval | domain-specific (not general RAG) | optional | — |
| **GerDaLIR** | German **legal** IR | OOD / domain | **diagnostic-only** (our policy) | — |

**Umbrella:** run the **MTEB(deu)** retrieval suite (Wehrli et al. 2024; tooling: `github.com/mayflower/mteb-de`)
for a standard, citeable, comparable score. Its retrieval core = GermanQuAD-Retrieval, GerDaLIR,
GermanDPR, XMarket.

### B. End-to-end / context-selection / RAG-answer benchmarks (for the full pipeline & a reranker)

| benchmark | what it is | use |
|---|---|---|
| **deutsche-telekom/Ger-RAG-eval** | 4×1,000 German-Wikipedia tasks: *choose-context-by-question*, *choose-question-by-context*, context↔question match, question↔answer match (LLM accuracy; CC-BY-SA-4.0) | the **choose-context** subtask is a hard **context-selection / reranking** eval — use it when a reranker exists |
| **GermanRAG** (`rasdani/germanrag`) | German RAG fine-tuning set (query, contexts, answer), **derived from GermanDPR** | RAG eval set — **leakage caution** (GermanDPR family) |
| **MEMERAG** | multilingual end-to-end RAG meta-eval (faithfulness/relevance) — *verify German coverage* | RAG answer-quality, not retrieval |

## Recommendation for our dense RAG embedder

1. **Add MIRACL (de) + mMARCO (de)** as primary RAG retrieval evals — open-domain + large-scale
   passage retrieval, far more RAG-representative than our FAQ/QA-only set, and the numbers users
   actually compare against.
2. **Add MLDR (de)** for long-document RAG and **WikipediaRetrievalMultilingual (de)** as a general
   German guardrail.
3. **Run the full MTEB(deu) retrieval suite** (via `mayflower/mteb-de`) for a standard, comparable
   score — this is what the published model card should eventually cite (we currently report "no
   MMTEB run").
4. Keep **GerDaLIR diagnostic-only** (legal, OOD). XMarket optional (e-commerce domain).
5. **For the (future) reranker:** add **Ger-RAG-eval** context-selection as a reranking eval.

## ⚠️ Leakage discipline (critical)

Our v6.x training includes `wiki_non_eval` (from v3 **DT-de-dpr**, the German-DPR/Wikipedia family).
**MIRACL (de), GermanDPR, WikipediaRetrievalMultilingual, and GermanRAG are all German-Wikipedia-based
and may overlap our training passages.** Before trusting any of these as eval, run
`filter_leakage_against_eval_texts` / the eval-leakage check and confirm train↔eval disjointness — and
keep them `eval_only: true, allowed_for_training: false` in `benchmarks/mteb_german_tasks.json`. WebFAQ
held-out and mMARCO (de) are the lowest-leakage-risk additions.

## Measured: MIRACL (de) reduced-corpus comparison (2026-06-17)

Method: full MIRACL-de corpus is **15.9M passages** (305 dev queries, 3,144 qrels) → ~11 GPU-h/model,
infeasible across models; the hard-negatives variant is script-broken on `datasets` 4.x. So a
**reduced corpus** = all relevant + **300k random distractors** (identical pool for every model). Fair
cross-model comparison; absolute scores are **higher than the official full-corpus leaderboard**.
(`outputs/v6-1-dense-top50/miracl_de_reduced_summary.json`.)

| model | params | R@10 | R@50 | R@100 | nDCG@10 | missing |
|---|--:|--:|--:|--:|--:|--:|
| Boldt-Embed-DE-350M (this) | 350M | 0.778 | 0.876 | 0.899 | **0.700** | 0.036 |
| multilingual-e5-base | 278M | 0.948 | 0.976 | 0.983 | 0.893 | 0.007 |
| multilingual-e5-large | 560M | 0.962 | 0.979 | 0.988 | 0.917 | 0.000 |
| bge-m3 | 568M | 0.957 | 0.984 | 0.985 | **0.920** | 0.003 |

**Finding:** on MIRACL (general open-domain German Wikipedia retrieval) this model **substantially
underperforms** the general multilingual models (nDCG@10 0.70 vs 0.89–0.92). It is a **WebFAQ/FAQ-RAG
specialist** (where it leads), **not** a general German retriever — MIRACL confirms this more starkly
than the GermanQuAD/DT-test guardrails. Note: MIRACL is German Wikipedia (potential overlap with our
`wiki_non_eval` training), yet the model still loses badly here — so no leakage advantage is visible;
if anything the gap is understated.

**mMARCO (de): NOT run.** The dataset is loading-script-based (broken on `datasets` 4.x), has no
German parquet mirror (`mteb/MMarco*` are Chinese; `mteb/MSMARCO` is English), and the collection is
8.8M passages. Running it needs converting `unicamp-dl/mmarco` + a full 8.8M index — deferred.

## Sources (checked 2026-06-17)

- MMTEB / MTEB(deu): https://arxiv.org/pdf/2502.13595 ; tooling https://github.com/mayflower/mteb-de ;
  https://huggingface.co/datasets/mteb/germanquad-retrieval
- MIRACL: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00595/117438/ ;
  https://aclanthology.org/2023.tacl-1.63.pdf
- Ger-RAG-eval: https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval
- GermanRAG: https://github.com/rasdani/germanrag
- MEMERAG: https://arxiv.org/pdf/2502.17163
- Multilingual RAG context: https://arxiv.org/pdf/2407.01463
