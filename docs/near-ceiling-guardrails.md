# Near-ceiling guardrail set

A held-out **near-ceiling** guardrail to validate the bounded rerank policy **without tuning on
GermanQuAD/DT-test**. GermanQuAD stays a guardrail; it must never become the policy-tuning target.
Pure stdlib, no ML.

- Module: `src/boldt_embed/near_ceiling_guardrail.py`
- CLI: `scripts/build_near_ceiling_guardrail.py`
- Artifact: `data/processed/v5/near_ceiling_guardrail.jsonl` (+ report)

## Definition

A **near-ceiling** query (all required): `first_stage_ndcg@10 >= 0.95`, `oracle_ndcg@10 >= 0.98`,
`positive_in_top_10`, `>= 20 candidates` (`>= 2 candidate sources` preferred — soft, reported). On
such queries the first stage is already (almost) perfect, so raw reranking can only churn them —
exactly where the bounded policy must prove it does no harm.

## Sources (non-public, leakage-safe)

Built from **held-out WebFAQ** (not used for training or policy fitting); held-out local RAG and a
private QA-passage split may be added if licensed and leakage-filtered. **GermanQuAD/DT-test are
excluded** and are never used to select policy thresholds.

- Public-eval exclusion is **label-based**: `--exclude-sources germanquad,dt_test` plus a fixed
  public set (`germanquad/dt_test/gerdalir/germandpr/miracl/mldr/sts22/mmteb`) matched against each
  row's `domain`/`source`/`query_id`/`candidate_source`. **Honest limitation:** raw GermanQuAD/DT
  lists that carry no such label (e.g. `gq…` ids, `domain: unknown`) are not auto-detected — so
  build the guardrail from the *intended* held-out source, not from raw public-eval dumps. Note this
  deliberately does **not** exclude `webfaq_heldout` — that held-out split is the guardrail *source*,
  not a public benchmark.
- **Train-disjoint is a HARD guard:** pass `--train-queries` (the training query ids); any overlap
  between the selected guardrail and training is a build **failure** (a guardrail must be disjoint).

## Built artifact (2026-06-15)

From `outputs/v5-small-rag/eval/conservative_scored/webfaq_scored.jsonl` (held-out WebFAQ):
**716 near-ceiling lists selected** (all first_stage nDCG@10 ≥ 1.0, oracle ≥ 1.0), 0 public-source
exclusions, **0 overlap with the 5,660 training queries** → status `pass`. Single-source (BM25) so
`multi_source_fraction = 0.0` (the ">=2 sources" criterion is soft). See
`outputs/v5-small-rag/near_ceiling_guardrail_report.json`.

## CLI

```bash
python scripts/build_near_ceiling_guardrail.py \
  --candidate-lists data/processed/v5/rag_candidate_lists.jsonl \
  --output data/processed/v5/near_ceiling_guardrail.jsonl \
  --report outputs/v5-small-rag/near_ceiling_guardrail_report.json \
  --target-size 1000 --exclude-sources germanquad,dt_test \
  --train-queries outputs/v5-small-rag/teacher/rag_train_scored.jsonl
```

`--dry-run` writes the report only (no torch). Report includes number selected, source/domain
distribution, first-stage/oracle nDCG distributions, candidate-source distribution, leakage-check
summary, and the training-overlap check.

## Acceptance

- ✅ The bounded policy can be validated on a **non-GermanQuAD** near-ceiling set (held-out WebFAQ,
  716 lists, train-disjoint).
- ✅ GermanQuAD remains a guardrail, **not** a tuning target — it is excluded from this set, and
  policy thresholds are never fit on it.
