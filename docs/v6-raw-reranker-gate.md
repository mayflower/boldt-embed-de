# v6 RAW reranker promotion gate

Decides whether the v6 reranker **MODEL** is promotable — judged as **raw lift over fixed candidate
lists**. **No bounded policy, no serving wrapper, no abstention.** A result evaluated in any policy
mode is rejected, and a model card that recommends a policy workaround fails the gate. Pure stdlib.

- Eval: `scripts/eval_v6_raw_reranker_lift.py` (`raw_lift_report`) — always emits `ranking_mode: raw`.
- Gate: `scripts/check_v6_raw_reranker_gate.py` (`raw_reranker_gate`).
- Outputs: `outputs/v6-reranker/eval/<set>_lift.json`, `outputs/v6-reranker/raw_gate.{json,md}`.

## Eval sets and metrics

| set | role | gated |
|---|---|---|
| `webfaq` | primary | yes — lift target |
| `local_rag` | primary (if present) | yes |
| `germanquad` | guardrail | yes — do-not-regress |
| `dt_test` | guardrail | yes — do-not-regress |
| `gerdalir` | diagnostic | **ignored** |

Per query: `first_stage_ndcg@10`, `raw_reranker_ndcg@10`, `delta_ndcg@10`, `mrr@10`,
`catastrophic_drop_rate` (per-query Δ ≤ −0.2), `positive_present_rate`, recall@10 (first vs reranked),
the **candidate-set-unchanged** sanity check (a reranker only reorders a fixed list — it must not add
or drop docs), and metrics by hardness bucket.

## Gate — pass only if

1. WebFAQ raw `delta_ndcg@10 ≥ +0.05`
2. local RAG raw `delta_ndcg@10 ≥ +0.03` (if present)
3. GermanQuAD raw `delta_ndcg@10 ≥ −0.005`
4. DT-test raw `delta_ndcg@10 ≥ −0.005`
5. GermanQuAD `catastrophic_drop_rate ≤ 0.03`
6. DT-test `catastrophic_drop_rate ≤ 0.02`
7. `positive_present_rate ≥ 0.8` on primary sets (so the eval is meaningful)
8. no public-eval leakage
9. **no policy-gated result used** — every gated report must have `ranking_mode == "raw"`

## Fail if

- Any result path / `ranking_mode` includes **`bounded`, `policy`, `abstain`** (or `margin_override`)
  as the evaluated ranking mode — rejected outright (`raw_only:<set>`).
- The reranker model card recommends a **policy workaround** (cross-checked via
  `validate_release_2026.check_no_policy_gated_recommendation`).

## Status: gate READY — real eval BLOCKED upstream (honest)

This gate is implemented and fixture-validated (`tests/test_v6_raw_reranker_gate.py`), but the **real
eval has not been run**, because the upstream **data** has not been built yet:

- **No trained v6 reranker.** `outputs/v6-reranker/checkpoints/boldt-rag-reranker-v6` does not exist.
- **No v6 candidate union lists.** `data/processed/v6/*_candidate_union.jsonl` do not exist — they
  require dense-v6 ∪ BM25 retrieval + Qwen3-Reranker-8B teacher scoring.

**Correction:** an earlier version of this doc said the eval was blocked because there was "no real
corpus on disk." That was **wrong** — the real WebFAQ eval (`outputs/v4-rag-reranker/eval/webfaq/`:
1,381 docs, 1,576 queries, qrels) is on disk, alongside GermanQuAD/DT-test/GerDaLIR corpora and a
BM25 index builder. Measured over the real WebFAQ corpus, the v6 dense retriever **materially fixes
recall** (Recall@100 0.651 → 0.964; `outputs/v6-dense-rag/webfaq_real_recall_bm25_vs_dense.json`), so
the recall precondition is **met**. The remaining gap is only building/teacher-scoring the v6
candidate union lists and training the raw reranker.

So **promotability is currently UNDECIDED** — the correct state: there is no raw reranker to evaluate
yet (and no policy-gated shortcut may substitute). The chain that must complete: build v6 candidate
union lists (teacher-scored) → train the v6 raw reranker → run this eval+gate.

## Run (once the inputs exist)

```bash
python scripts/eval_v6_raw_reranker_lift.py \
  --reranker outputs/v6-reranker/checkpoints/boldt-rag-reranker-v6 \
  --candidate-lists data/processed/v6/webfaq_eval_candidate_union.jsonl \
  --output outputs/v6-reranker/eval/webfaq_lift.json
# (repeat for germanquad/dt_test/local_rag/gerdalir)
python scripts/check_v6_raw_reranker_gate.py \
  --eval-dir outputs/v6-reranker/eval \
  --output outputs/v6-reranker/raw_gate.json \
  --markdown outputs/v6-reranker/raw_gate.md
```

## Acceptance

- ✅ This gate decides whether the reranker **model itself** is promotable (raw lift), with no policy
  shortcut admissible — a policy-mode result is rejected and a policy-recommending card fails.
- ⏳ The decision is **pending real inputs** (trained v6 reranker + v6 candidate union lists); the gate
  is ready to render the verdict the moment they exist.
