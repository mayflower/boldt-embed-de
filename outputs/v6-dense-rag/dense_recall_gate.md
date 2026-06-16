# Dense-recall STOP gate: **fail (ADVISORY — not blocking)**

_A reranker cannot rank a document the first stage never retrieved. This gate blocks reranker
training only when positives are genuinely ABSENT. See `scripts/check_dense_recall_gate.py` and
`docs/dense-recall-gate.md`._

| check | value | target | status |
|---|--:|--:|---|
| WebFAQ Recall@100 | 0.9638 | 0.95 | ✅ |
| WebFAQ positive_in_top_50 | 0.8832 | 0.90 | ❌ |
| WebFAQ oracle_ndcg@10 | 0.9657 | 0.95 | ✅ |
| missing_positive_rate | 0.0343 | ≤ 0.10 | ✅ |
| candidate_union_size | 200 | ≥ 20 | ✅ |

- **positives_absent: false** — Recall@100 0.964 and oracle 0.966 pass; the positives the reranker
  needs ARE present (missing-positive rate 0.349 → 0.036 vs BM25).
- **blocking: false, stop_file_written: false** — the only miss is the strict top-50 *ranking* target
  (0.883 < 0.90), so the verdict is **advisory**: no `STOP_RERANKER_TRAINING.md` is written and
  reranker training is **not** blocked.
- For a reranker that reranks the full 200-candidate union (present 0.966), Recall@100 / present-rate
  is the binding recall metric, not top-50 — so the dense retriever's recall is sufficient.

**Dense embedder is NOT recommended yet** (the dense gate must reach a clean pass; top-50 0.883 <
0.90). This is a retrieval-quality target, not a missing-positive (absence) condition. See
`outputs/v6-dense-rag/dense_recall_gate.json`.
