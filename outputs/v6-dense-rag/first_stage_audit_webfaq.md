# First-stage recall audit — webfaq

**Bottleneck: `first_stage_recall`.** 34.5% of positives were never retrieved — the reranker cannot see them. Fix retrieval / candidate-list construction, not the reranker.

_1360 queries, 1360 positives. A reranker can only reorder what the first stage retrieved — injected/oracle sources (e.g. `manual`) are NOT counted as retriever hits._

## Recall & nDCG

| metric | value |
|---|--:|
| recall@10 | 0.650735 |
| recall@20 | 0.655147 |
| recall@50 | 0.655147 |
| recall@100 | 0.655147 |
| recall@200 | 0.655147 |
| positive_in_top_10 rate | 0.650735 |
| first-stage nDCG@10 | 0.596869 |
| oracle nDCG@10 (retriever-only, realistic) | 0.655147 |
| oracle nDCG@10 (with injected positives) | 1.0 |
| **upper-bound reranker lift (realistic)** | **0.058278** |
| upper-bound lift IF candidate set were perfect | 0.403131 |
| illusory lift from injected positives | 0.344853 |

## Missing positives (reranker can NEVER recover these)

- missing_positive_count: **469** (rate **0.344853**)
- by domain: {'faq_real': 469}
- by query_style: {'faq_real': 469}
- by source-of-missing: {'injected_only': 469}

## Candidate-source contribution

| bucket | positives |
|---|--:|
| BM25 hits | 891 |
| dense hits | 0 |
| other-retriever hits | 0 |
| BM25-only | 891 |
| dense-only | 0 |
| overlap (BM25 ∧ dense) | 0 |
| union (any retriever) | 891 |
| injected-only (NOT retrieved) | 469 |
| absent from list | 0 |

**Has a dense first stage in these lists: False.** No dense candidate source is present — these lists are BM25-only (plus injected gold), so dense-vs-BM25 recall cannot be compared from this data.

## Examples (deterministic)

- `qba7287681818122a` [faq_real] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `df6b10f9985e70c83`: “Wie lange dauert eine Autofahrt von Sitten nach Montreux?”
- `qcedd7101273eec77` [faq_real] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `df392ffd87bc01904`: “Wie lange dauert eine Autofahrt von Steinpoint nach Wiesing?”
- `q67b44a2202451251` [faq_real] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `d63dadd98cd4121a8`: “Wie viel kostet es, von Leoben nach Edlach zu fahren?”
- `qa84deaa17effa36d` [faq_real] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `d538eb30fc4848a28`: “Wieviel kostet der Aufenthalt in der Hotel Wuzhen wangjinli Boutique?”
- `q350cf3c26129e8ed` [faq_real] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `d74e539c79b9a3e2f`: “Wie viele Stunden ist Rüdtligen von Boll mit dem Flugzeug entfernt?”
- `q8e4ba7240914ba0e` [faq_real] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `de3af4ddc3c748430`: “Wie viel kostet es, von Wiesing nach Völs zu fahren?”
- `q42ab4f4419486265` [faq_real] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `dbd6fde05314f1076`: “Wie viel kostet es, von Inning nach Maria Ellend zu fahren?”
- `q7c5f23ab255d7e91` [faq_real] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `d45008a580bc513a0`: “Wie lange dauert ein Hubschrauberflug von Treglwang nach Schwanberg?”
