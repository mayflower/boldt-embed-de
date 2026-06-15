# Hardness-aware RAG reranker evaluation

Stops over-penalizing the reranker on **near-ceiling** candidate lists while still blocking real
degradation. Pure stdlib, no ML.

- Module: `src/boldt_embed/hardness_aware_eval.py`
- CLIs: `scripts/analyze_candidate_list_hardness.py`, `scripts/eval_v5_rag_lift.py`

## Why

In v4, GermanQuAD/DT-test first stages were near-ceiling (recall ≈ 0.96–0.99, oracle ≈ 1.0). A
reranker can only **churn** an already-correct top order there, so its small negative delta
(−0.0007 to −0.07) read as "failure" and blocked promotion — even though WebFAQ lifted +0.29.
That is the wrong test. v5 evaluates by **difficulty bucket** and uses near-ceiling sets as
**guardrails**, not primary promotion signals.

## Per-list metrics

For each fixed candidate list: `first_stage_ndcg@10`, `oracle_ndcg@10`, `positive_in_top_10`,
`positive_in_top_50`, `num_candidates`, `num_candidate_sources`, and a `hardness_bucket`.

## Hardness buckets

| bucket | rule | meaning |
|---|---|---|
| `no_room` | oracle ≥ 0.98 and first_stage ≥ 0.95 | near-ceiling; reranking can only churn |
| `easy` | first_stage ≥ 0.85 | already strong |
| `medium` | 0.5 ≤ first_stage < 0.85 | real headroom |
| `hard` | first_stage < 0.5 and oracle ≥ 0.8 | recall present, order bad — best case for a reranker |
| `impossible` | oracle < 0.8 | positive not (well) in the candidate set — reranking cannot fix |

Precedence: `impossible` (oracle < 0.8) and `no_room` are checked before `easy`/`medium`/`hard`.

## Evaluation policy

1. **Primary promotion** uses the **medium+hard** buckets on primary sets (WebFAQ / local /
   private RAG) — the only buckets where a reranker can meaningfully help.
2. **GermanQuAD / DT-test are do-not-regress guardrails**, never a primary signal.
3. A set that is **mostly `no_room`** gets a small negative-delta tolerance of **−0.005**, unless
   per-query **catastrophic drops** exceed the max rate. Guardrail sets that still have headroom
   are held to neutral-or-better (0.0).
4. Lift is reported **macro** (mean of per-bucket mean deltas) and **micro** (mean per-query delta)
   per bucket.
5. Per-query **catastrophic drops** (a single query's delta ≤ −0.2) are counted and rate-limited.

## Gate

Promotion passes iff **all** hold:

- every primary set has **medium+hard** lift strictly positive (micro and macro), and has at least
  one medium/hard query to establish it;
- no guardrail set falls below its tolerance (−0.005 if near-ceiling, else 0.0);
- no set exceeds the max per-query **catastrophic-drop rate** (default 5%);
- at least one primary set is present.

## CLIs

```
# difficulty distribution of a candidate-list set (no reranker needed)
python scripts/analyze_candidate_list_hardness.py \
  --candidate-lists data/processed/v5/webfaq_eval_lists.jsonl --eval-set webfaq \
  --output outputs/v5-small-rag/hardness_webfaq.json

# hardness-aware lift + promotion gate
python scripts/eval_v5_rag_lift.py \
  --primary webfaq=…/webfaq_lists.jsonl local_rag=…/local_lists.jsonl \
  --guardrail germanquad=…/gq_lists.jsonl dt_test=…/dt_lists.jsonl \
  --report outputs/v5-small-rag/v5_rag_lift_gate.json
```

Reranked order comes from per-candidate `reranker_score` (fallback `teacher_score`) for fixed
pre-scored lists, or from a real `--reranker` checkpoint (lazy ML). `--dry-run` (no `--reranker`)
imports no torch.

## Acceptance

- Reranker promotion no longer depends on impossible near-ceiling improvements — it is driven by
  medium+hard headroom on primary sets.
- Real degradation is still blocked — guardrail tolerance is small (−0.005 near-ceiling, 0.0
  otherwise) and per-query catastrophic drops are rate-limited.
