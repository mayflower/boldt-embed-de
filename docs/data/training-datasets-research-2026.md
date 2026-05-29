# German Embedding Training Datasets — Research (2026-05-29)

Goal: identify **useful, permissively-licensed German training data that is disjoint from the
evaluation benchmarks**, so MMTEB-German / GermanQuAD results measure generalization, not
in-domain memorization. (Facts below verified from HF dataset cards on 2026-05-29.)

> **Why this matters:** the earlier GermanQuAD run trained on GermanQuAD-train and scored on
> GermanQuAD-test. That is a valid held-out split but **in-domain**, and GermanQuAD *is* an
> MMTEB task — so it must become **eval-only**. Training data must come from elsewhere.

## Candidate training datasets (NON-benchmark, permissive)

| Dataset (HF id) | Type | Lang/size | License | Source domain | Verdict |
|---|---|---|---|---|---|
| `unicamp-dl/mmarco` (`german`) | passage ranking (q→passage) | de, 93.8 GB | **Apache-2.0** (card; verify MS-MARCO upstream) | **web (MS MARCO)** | ✅ **Top pick for training** — non-Wikipedia → disjoint from GermanQuAD/GerDaLIR |
| `clips/mqa` (`de`) | FAQ/CQA (question→answer) | de, part of 100M–1B | **CC0-1.0** | web (Common Crawl) | ✅ cleanest license; web domain |
| `nthakur/swim-ir-monolingual` (`de`) | synthetic q→passage | de, **447k** | **CC-BY-SA-4.0** | **Wikipedia** (via MIRACL) | ◑ good, but ⚠️ overlaps MIRACL **and** GermanQuAD (Wikipedia) — dedup or don't eval on those |
| German Wikipedia (e.g. `wikimedia/wikipedia` `20231101.de`) | corpus for synthetic / mined pairs | de, millions | **CC-BY-SA-4.0** | Wikipedia | ◑ clean corpus; same Wikipedia-overlap caveat |
| Synthetic from FineWeb-2 (de) | generated q→passage (our pipeline) | de, scalable | `synthetic` | web (base-model domain) | ✅ clean; matches base pretraining domain |

### Rejected / eval-only (do NOT train on these)
- **GermanQuAD / GermanDPR**, **GerDaLIR(Small)**, **MLDR(de)**, **MIRACL(de)**, **STS22(de)**,
  **MassiveIntentClassification(de)**, **PawsX(de)**, **XNLI(de)**, German clustering benchmark
  (arXiv 2401.02709) — all are MMTEB/German eval tasks → **held out** (ADR-005/009).
- **XNLI(de)**: also typically **CC-BY-NC** (non-commercial) → unsuitable for a permissive release.

## Train ↔ Eval separation matrix (corpus-domain disjointness)

| Train source (domain) | GermanQuAD (Wikipedia) | GerDaLIR (legal) | MIRACL-de (Wikipedia) |
|---|---|---|---|
| mMARCO-de (web) | ✅ disjoint | ✅ disjoint | ✅ disjoint |
| clips/mqa (web FAQ) | ✅ disjoint | ✅ disjoint | ✅ disjoint |
| SWIM-IR-de (Wikipedia) | ⚠️ overlap (dedup) | ✅ disjoint | ❌ same corpus — don't eval |
| Wikipedia-mined (Wikipedia) | ⚠️ overlap (dedup) | ✅ disjoint | ⚠️ overlap |

**Cleanest honest setup:** train on **mMARCO-de + clips/mqa** (web), evaluate on **GermanQuAD +
GerDaLIR** (Wikipedia + legal). Train and eval corpora are different sources → genuine
generalization. Add SWIM-IR only with passage-level dedup against the eval corpora, and then
do **not** report MIRACL.

## Hard negatives & loss
Mine hard negatives per query with BM25 + the in-training embedder (and later the reranker) —
mMARCO ships BM25 negatives; `boldt_embed.reranker.mine_hard_negatives` covers the rest.
Loss: MNRL/InfoNCE with in-batch + mined hard negatives (ADR-002), Matryoshka wrapper (ADR-007).

## Mandatory leakage control before any benchmark claim
Run `boldt_embed.data.find_leakage(train_records, eval_corpus_texts)` (exact + token-Jaccard)
for every (training set × eval corpus) pair and drop hits. This is the gate that makes the
GermanQuAD/MMTEB numbers honest (ADR-009, `docs/data/leakage-policy.md`).

## Recommended next run (replaces the in-domain GermanQuAD run)
1. Stream `unicamp-dl/mmarco` `german` → build (query, positive_passage) + BM25 hard negatives.
2. Optionally add `clips/mqa` `de` (question→accepted answer).
3. Dedup/leakage-check against GermanQuAD + GerDaLIR eval corpora.
4. Train causal + bidirectional (MNRL + Matryoshka) on the A6000.
5. Evaluate on **held-out** GermanQuAD + GerDaLIR (and MMTEB-de) — a true generalization number.

## Sources (fetched 2026-05-29)
- clips/mqa — https://huggingface.co/datasets/clips/mqa (CC0-1.0; de; FAQ/CQA from Common Crawl)
- SWIM-IR — https://huggingface.co/datasets/nthakur/swim-ir-monolingual (CC-BY-SA-4.0; de 447k) · paper arXiv 2311.05800 (NAACL'24)
- mMARCO — https://huggingface.co/datasets/unicamp-dl/mmarco (Apache-2.0 per card; `german` config)
- deepset-mxbai-embed-de-large-v1 — https://huggingface.co/mixedbread-ai/deepset-mxbai-embed-de-large-v1 (30M+ German pairs, explicit train/test de-overlap)
- Multilingual-E5 technical report — arXiv 2402.05672 · German clustering benchmark — arXiv 2401.02709
