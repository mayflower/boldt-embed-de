# v4 RAG reranker eval + promotion gate

The v4 reranker is judged on **lift over FIXED candidate lists** — does it promote
answer-supporting passages inside a real top-k? — and promotion reflects **RAG usefulness, not
legal transfer**. Module: `boldt_embed.rag_reranker_eval` · lift CLI:
`scripts/eval_rag_reranker_lift.py` · gate CLI: `scripts/check_rag_reranker_promotion_gate.py`.

## Inputs

Fixed candidate lists (one per eval set; see `docs/rag-candidate-lists.md`): **WebFAQ held-out**,
**GermanQuAD**, **DT-test**, **local RAG**, plus optional **GerDaLIR** (diagnostic). A reranker
checkpoint (real run) and optional teacher reranker scores.

## Metrics (per eval set, over the fixed list)

`first_stage_ndcg@10`, `reranked_ndcg@10`, **`delta_ndcg@10`**, `first_stage_mrr@10`,
`reranked_mrr@10`, `positive_in_top_10` before/after, `answer_support_at_10`, `oracle_ndcg@10`,
and `first_stage_recall_top_10` (how often a positive is even in the fixed list — reranking can
only help when it is). `fixed_candidates` is true only if every query carried a candidate list.

```bash
python scripts/eval_rag_reranker_lift.py --reranker <ckpt> \
  --candidate-lists data/processed/v4/rag_reranker_eval_webfaq_lists.jsonl \
  --output outputs/v4-rag-reranker/eval/reranker_lift_webfaq.json
# repeat for germanquad / dt_test / local_rag; add --diagnostic for gerdalir
```

## Promotion gate

`check_rag_reranker_promotion_gate.py --eval-dir <dir of reranker_lift_*.json>` PASSES only if:

- **WebFAQ held-out** `delta_ndcg@10 ≥ 0.03` and **local_rag** `≥ 0.03` (if present),
- **GermanQuAD** and **DT-test** `delta_ndcg@10 ≥ 0.0` (neutral-or-better),
- **no eval set** drops more than `-0.02` (catastrophic),
- first-stage recall is high enough (`first_stage_recall_top_10 ≥ 0.5`) for reranking to be
  meaningful,
- every reported (non-diagnostic) set uses **fixed candidate lists**.

**GerDaLIR / legal are DIAGNOSTIC ONLY** — reported (and may show a negative delta) but **never
gate** v4 promotion.

```bash
python scripts/check_rag_reranker_promotion_gate.py \
  --eval-dir outputs/v4-rag-reranker/eval \
  --output outputs/v4-rag-reranker/eval/rag_reranker_gate.json \
  --markdown outputs/v4-rag-reranker/eval/rag_reranker_gate.md
# exit 0 = promotable; exit 1 = do not promote.
```

So the reranker can only be promoted if it genuinely lifts RAG/FAQ retrieval **and** does not
degrade GermanQuAD/DT-test — and a bad legal (GerDaLIR) number can neither block nor enable it.
`--dry-run` on the lift CLI imports no torch (reranks by any precomputed score) for CI checks.
