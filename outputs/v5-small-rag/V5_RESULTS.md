# v5 small-RAG reranker — results (prompt-4 real run)

## Verdict: **fail / not_promoted**  (gate: fail)

**Status: Experimental; not recommended for production reranking.** Next step: rerank-or-abstain calibration on near-ceiling first-stage lists.

## Training data (real, leakage-filtered vs DT-test + GermanQuAD)
- raw rows: 6500 ({'faq_real': 2000, 'qa_passage_non_eval': 2500, 'german_stress': 1200, 'long_doc_chunks': 800})
- candidate lists: 5660 (BM25 recall 0.8708)
- teacher-scored pairs: 113145 (Qwen3-Reranker-8B); gold/hardneg/uncertain 5660/104091/3394
- model: Boldt/Boldt-DC-350M, loss listwise_kl+pairwise+pointwise_confident, FAQ share 0.217 (not_faq_only=True)
- domains with no real source (omitted, NOT faked): ['web_nonfaq', 'local_rag']

## Hardness-aware gate (nDCG@10 over FIXED candidate lists)

| eval set | role | overall delta | medium+hard | no_room | catastrophic | result |
|---|---|--:|--:|--:|--:|:--|
| webfaq | primary | +0.166548 | +0.369936 | 0.539706 | 0.010294 | pass |
| germanquad | guardrail | -0.028452 | +0.34552 | 0.842 | 0.169333 | FAIL |
| dt_test | guardrail | +0.021118 | +0.541478 | 0.961 | 0.0 | pass |

## v4 -> v5 on the same fixed guardrail lists
- GermanQuAD: -0.0711 -> -0.028452 (degradation reduced, still fails)
- DT-test: -0.0007 -> 0.021118 (now positive)

## Conservative reranker (rank-preservation loss) — real measured progress

| approach | GermanQuAD overall | GermanQuAD catastrophic | WebFAQ overall | DT-test overall | gate |
|---|--:|--:|--:|--:|:--|
| raw_v5 | -0.028452 | 0.169333 | 0.166548 | 0.021118 | fail |
| abstain_only | -0.001553 | 0.103333 | 0.128456 | 0.01799 | fail |
| conservative_only | 0.009421 | 0.122 | 0.137925 | 0.021205 | fail |
| conservative_plus_abstain | 0.024292 | 0.074 | 0.097458 | 0.019332 | fail |

_Conservative-only gate fails: ['germanquad_catastrophic_rate']. Conservative+abstain gate fails: ['germanquad_catastrophic', 'dt_test_beats_always_rerank']. NOT promoted._

Real measured progress. Conservative training (rank-preservation penalty on high-first-stage-confidence lists) reduces near-ceiling churn: GermanQuAD overall -0.0285 -> +0.0094 (conservative) / +0.0243 (conservative+abstain) and catastrophic 0.169 -> 0.122 / 0.074, while WebFAQ (+0.0975) and DT-test (+0.0193) stay healthy. NOT promoted: the remaining failure is catastrophic tail risk on near-ceiling GermanQuAD lists (0.074 > 0.03 bar). Next step: a bounded / top-preserving rerank policy.

## Interpretation (raw v5)

v5 is better than v4 but still NOT promotable. Multi-domain training (FAQ share 0.217) lifts every set strongly where there is real headroom (medium+hard buckets), and both guardrails improved over v4 (GermanQuAD -0.0711->-0.0285, DT-test -0.0007->+0.0211). But the gate FAILS: on GermanQuAD the reranker over-reorders near-ceiling first-stage lists (84% no_room), netting -0.0285 and 16.9% catastrophic per-query drops. Next step: rerank-or-abstain calibration on confident first stages.

