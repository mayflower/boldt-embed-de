# First-stage recall audit (v6 gate — run before any reranker training)

**A reranker can only reorder documents the first stage retrieved.** v5's failure analysis traced the
WebFAQ under-lift to `no_useful_first_stage_score` — positives missing from candidate lists. This
audit quantifies that, per eval set, and names the bottleneck: **dense retrieval, BM25 retrieval,
candidate-list construction, or reranker quality.**

- Core: `src/boldt_embed/first_stage_audit.py` — CLI: `scripts/audit_first_stage_recall.py`
- Reports: `outputs/v6-dense-rag/first_stage_audit_{webfaq,germanquad,dt_test}.{json,md}`

**Honesty rule:** injected/oracle candidate sources (`manual`/`gold`/…) are **not** retriever hits.
The v5 pipeline *injects* the gold positive as a `manual` candidate when the retriever misses it.
Recall and the realistic "perfect-reranker" ceiling are computed over the **retriever** set only —
counting injected positives is what produced v5's illusory +0.14 "lift".

## Result (real run, 2026-06-15)

| eval set | n | recall@10 | recall@20 | first-stage nDCG@10 | oracle (retriever) | **realistic reranker ceiling** | illusory (injection) | missing | **bottleneck** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| **WebFAQ** | 1360 | **0.651** | 0.655 | 0.597 | 0.655 | **+0.058** | +0.345 | **34.5%** | **first-stage recall** |
| GermanQuAD | 1500 | 0.961 | 0.975 | 0.906 | 0.975 | +0.070 | +0.025 | 2.5% | reranker quality |
| DT-test | 1000 | 0.992 | 0.994 | 0.977 | 0.994 | +0.017 | +0.006 | 0.6% | near-ceiling |

## What this says about the bottleneck

- **WebFAQ → first-stage RECALL / candidate-list construction (NOT the reranker).** 34.5% of WebFAQ
  positives (469/1360) are **never retrieved** — present only as injected `manual` candidates. Even a
  **perfect** reranker over the actual candidate set can add only **+0.058** nDCG@10 (oracle 0.655 −
  first-stage 0.597). The +0.345 "ceiling with injected positives" is **illusory**: it credits gold
  documents the first stage never surfaced. This is exactly the v5 trap — raw reranking appeared to
  lift WebFAQ +0.14 only because the positive had been injected.
- **GermanQuAD → reranker QUALITY.** 96.1% recall@10; positives are retrieved but ranked low, leaving
  a **realistic +0.070** headroom a better reranker could capture. This is the one set where reranker
  training is the right lever.
- **DT-test → near-ceiling.** 99.2% recall@10, first stage already 0.977; reranking only churns.

## Two structural findings (the real story)

1. **There is NO dense first stage in these lists — they are BM25 top-20 only** (`has_dense_source =
   false` on every set). The only retriever source is `bm25`; `manual` is injected gold. So we
   **cannot** compare dense-vs-BM25 recall from this data, and the obvious fix for WebFAQ recall —
   adding a dense retriever to candidate construction — **has never been tried** in these eval lists.
2. **Recall@k for k > 20 is unmeasurable here.** The candidate lists are truncated at BM25 top-20
   (candidate count is 20, or 21 when the positive is injected), so recall@20/50/100/200 all plateau
   at the top-20 value. Whether deeper BM25 (top-100/200) or a dense retriever recovers the missing
   34.5% **requires re-running retrieval** — the v6 first step.

The missing WebFAQ positives are **entity-heavy, paraphrased FAQ** ("Wie lange dauert eine Autofahrt
von Sitten nach Montreux?", "Wie viel kostet es, von Inning nach Maria Ellend zu fahren?") — queries
whose answer passages share few surface tokens with the query, exactly where lexical BM25 fails and a
dense retriever is expected to help.

## Decision

**Do not train another reranker for WebFAQ yet.** The WebFAQ bottleneck is retrieval recall, not
reranking — the realistic reranker ceiling on the current BM25-only lists is only +0.058. The v6
priorities follow directly:

1. **Build a dense first stage** (Boldt dense v3/v5 + e5-base + Qwen3-Embedding-0.6B; teacher 8B as
   ceiling) and **re-run this audit** with real dense candidate sources, so recall@k and BM25/dense
   overlap can finally be measured.
2. **Re-do candidate-list construction at greater depth** (top-100/200, BM25 ∪ dense) so positives
   are actually present.
3. **Then** train a standalone reranker — first on **GermanQuAD-style** lists, the one set with real,
   retrieved, low-ranked headroom (+0.070).

## Reproduce

```bash
python scripts/audit_first_stage_recall.py \
  --eval-set webfaq \
  --candidate-lists outputs/v5-small-rag/eval/conservative_scored/webfaq_scored.jsonl \
  --output outputs/v6-dense-rag/first_stage_audit_webfaq.json \
  --markdown outputs/v6-dense-rag/first_stage_audit_webfaq.md
# (optional) --queries/--qrels/--corpus to supply gold + example text from separate files
```

## Addendum (2026-06-15) — dense-v6 recall over the REAL WebFAQ corpus

Step 1 above has now been run against the real WebFAQ eval corpus that is on disk all along
(`outputs/v4-rag-reranker/eval/webfaq/`: 1,381 docs, 1,576 queries, qrels). BM25 vs the trained
dense Boldt-v6 retriever, recall over the real corpus
(`outputs/v6-dense-rag/webfaq_real_recall_bm25_vs_dense.json`):

| retriever | recall@10 | recall@20 | recall@50 | **recall@100** | nDCG@10 |
|---|--:|--:|--:|--:|--:|
| BM25 | 0.638 | 0.642 | 0.647 | **0.651** | 0.586 |
| dense Boldt-v6 | 0.739 | 0.791 | 0.883 | **0.964** | 0.671 |

**Dense v6 materially fixes the recall bottleneck: Recall@100 0.651 → 0.964 (+0.313); missing-positive
rate 0.349 → 0.036.** So the v6-reranker precondition ("candidate recall is fixed") is **met**. Next:
build BM25 ∪ dense-v6 candidate union lists, teacher-score them, and train the raw reranker.
