# Teacher-threshold calibration

Status: **PASS**

- embedder threshold: 2.0 → kept 22736 (median rerank 7.1875)
- reranker threshold: 4.0 → kept 21676 (median rerank 7.25)
- suspicious positive rate: 0.0818 · unknown-license rows: 0

## Acceptance by threshold (positives)

| threshold | accepted | rate |
|--:|--:|--:|
| -2 | 24397 | 0.9853 |
| 0 | 23235 | 0.9384 |
| 1 | 22884 | 0.9242 |
| 2 | 22736 | 0.9182 |
| 3 | 22380 | 0.9038 |
| 4 | 21676 | 0.8754 |
| 5 | 20804 | 0.8402 |

## Embedder vs reranker accepted by domain

| domain | embedder | reranker |
|---|--:|--:|
| faq_real | 4248 | 3520 |
| german_stress | 750 | 733 |
| web | 9879 | 9684 |
| wiki_non_eval | 7859 | 7739 |

## Low-score positives (suspicious)
- [faq_real] score=-8.5: Wo darf ich mit dem Wohnmobil übernachten?
- [faq_real] score=-8.1875: Thema am 16. August war
- [faq_real] score=-7.78125: Themen am 2. Oktober waren
- [faq_real] score=-7.40625: HS | 12.10. - 25.10. | Herbstferien
- [wiki_non_eval] score=-7.3125: Wo wurde Anton Höfle geboren?
