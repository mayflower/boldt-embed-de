# v5 catastrophic GermanQuAD drop analysis

Answers the open question after bounded reranking still failed the *fitted* gate: are the remaining
catastrophic GermanQuAD drops **policy-fixable** (bounded reranking) or **data/model-fixable** (the
reranker genuinely prefers the wrong document)? No training — pure analysis of existing scores.

- Module: `src/boldt_embed/rerank_error_analysis.py`
- CLI: `scripts/analyze_catastrophic_rerank_drops.py`
- Output: `outputs/v5-small-rag/bounded/catastrophic_analysis.{json,md}`

## Method

A query is a **catastrophic drop** when reranking (always-rerank over the conservative scores)
loses ≥ 0.2 nDCG@10 vs the first stage. For each, we record first-stage/reranked top-10, the
positive's initial→final rank, gap features, source mix, rank displacements; classify the error
type; and test whether each bounded policy (top1_lock, top3_lock, bounded_downshift D=1,
blend α=0.85, margin_override M=3) turns it back into a non-catastrophic result. Qrels are used for
**analysis only** — never by any policy decision.

## Result (EXECUTED 2026-06-15, GermanQuAD conservative scores)

**185 of 1500 queries catastrophic. 100% are policy-fixable; 0 require data/model changes.**

| error type | count |
|---|--:|
| query_style_mismatch (short factoid query vs long Wikipedia passage) | 135 |
| reranker_promotes_longer_but_wrong_doc | 26 |
| insufficient_first_stage_features (tiny first-stage gap) | 9 |
| lexical_exact_positive_demoted | 8 |
| positive_demoted_from_top1_or_top3 | 6 |
| unknown | 1 |

Average first-stage gap and reranker gap are reported in the JSON. Fixability (of the 185):

| bounded policy | fixes |
|---|--:|
| top3_lock | **183** |
| blend α=0.85 | 181 |
| top1_lock | 172 |
| margin_override M=3 | 167 |
| bounded_downshift D=1 | 15 |

## Conclusion

**The remaining failures are POLICY-fixable, not data/model-fixable.** Every catastrophic drop is
fixed by at least one bounded policy (top3_lock alone fixes 183/185 = 98.9%). The dominant pattern
(`query_style_mismatch`, 135) is the reranker demoting a *correct long passage* on a short factoid
query — a **reordering** error on a near-ceiling first stage, not a retrieval failure. `bounded_downshift`
barely helps (15) because it bounds how far *other* docs drop, not whether the confident top-k is
preserved — confirming that **top-preserving** policies (top-k lock / blend / margin_override) are
the right family.

This matches the gate finding: unconditional `margin_override` passes the full 7-check gate
(GermanQuAD catastrophic 0.0147). **No new checkpoint is warranted for these drops** — they are
deployment-policy fixable. The open work is *selection* (a WebFAQ-dev threshold did not transfer;
an unconditional top-preserving policy does) and validating it on a held-out near-ceiling set.

## Acceptance

- ✅ We now know the remaining failures are **policy-fixable** (100% fixed by some bounded policy;
  top3_lock 98.9%), so the next step is a deployment policy decision, **not** retraining.
