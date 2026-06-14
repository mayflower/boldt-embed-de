# v3 real-domain experiment — EXECUTED (2026-06-14, RTX A6000)

Verdict: **invalid_for_promotion** — domain-quality gate fail: admin_real + legal_adjacency_real absent; faq_real accepted 4248 < 5000 floor; effective web+wiki share too high. Real FAQ added, but the 7-domain target is not met.

## Key finding
REAL FAQ (WebFAQ, CC-BY-4.0) passes teacher validation at 70.8% (>=2.0) / 58.7% (>=4.0) vs v2 SYNTHETIC faq at 5.7%. The v2 failure was the synthetic queries, not the FAQ domain.

| domain | teacher accept >=2.0 |
|---|---:|
| faq_real (REAL WebFAQ) | 70.8% |
| web | 98.8% |
| wiki_non_eval | 98.2% |
| german_stress | 98.6% |

(v2 synthetic faq: 5.7%.)

## Dense retrieval nDCG@10 (same harness as v1/v2)
| set | base | v1 | v2 | v3 | e5-base |
|---|--:|--:|--:|--:|--:|
| GermanQuAD | 0.288 | 0.883 | 0.886 | **0.885** | 0.939 |
| DT-test | 0.223 | 0.950 | 0.944 | **0.970** | 0.994 |
| GerDaLIR (legal OOD) | 0.003 | 0.078 | 0.110 | **0.089** | 0.153 |

## Honest reading
- DT-test 0.970 = best causal yet (v1 0.950, v2 0.944) — real FAQ + cleaner teacher-validated data helped in-domain.
- GermanQuAD 0.885 flat (within noise of v1/v2).
- GerDaLIR (legal OOD) 0.089 < v2 0.110: v3 dropped v2's synthetic legal-adjacent data and added FAQ instead; FAQ does not transfer to legal. Legal needs REAL legal pairs (still unsourced).

Real data: 24,761 candidates (faq 6000 REAL, web 10000, wiki 8000, stress 761); 0 leakage hits;
embedder trained on 22,736 teacher-validated positives. admin/legal still unsourced → not promotable.
