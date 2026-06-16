# First-stage recall audit — germanquad

**Bottleneck: `reranker_quality`.** positives are retrieved but ranked low; a better reranker could realistically lift nDCG@10 by up to 0.069555.

_1500 queries, 1500 positives. A reranker can only reorder what the first stage retrieved — injected/oracle sources (e.g. `manual`) are NOT counted as retriever hits._

## Recall & nDCG

| metric | value |
|---|--:|
| recall@10 | 0.961333 |
| recall@20 | 0.975333 |
| recall@50 | 0.975333 |
| recall@100 | 0.975333 |
| recall@200 | 0.975333 |
| positive_in_top_10 rate | 0.961333 |
| first-stage nDCG@10 | 0.905778 |
| oracle nDCG@10 (retriever-only, realistic) | 0.975333 |
| oracle nDCG@10 (with injected positives) | 1.0 |
| **upper-bound reranker lift (realistic)** | **0.069555** |
| upper-bound lift IF candidate set were perfect | 0.094222 |
| illusory lift from injected positives | 0.024667 |

## Missing positives (reranker can NEVER recover these)

- missing_positive_count: **37** (rate **0.024667**)
- by domain: {'unknown': 37}
- by query_style: {'unknown': 37}
- by source-of-missing: {'injected_only': 37}

## Candidate-source contribution

| bucket | positives |
|---|--:|
| BM25 hits | 1463 |
| dense hits | 0 |
| other-retriever hits | 0 |
| BM25-only | 1463 |
| dense-only | 0 |
| overlap (BM25 ∧ dense) | 0 |
| union (any retriever) | 1463 |
| injected-only (NOT retrieved) | 37 |
| absent from list | 0 |

**Has a dense first stage in these lists: False.** No dense candidate source is present — these lists are BM25-only (plus injected gold), so dense-vs-BM25 recall cannot be compared from this data.

## Examples (deterministic)

- `gq650` [unknown] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `g137`: “Wie heißen die Lehrer an der Uni?”
- `gq870` [unknown] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `g171`: “Aus welcher Epoche stammt das erste nachgewiesene kultivierte Getreide?”
- `gq176` [unknown] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `g42`: “Woher kommen die meisten ausländischen Touristen Berns?”
- `gq1682` [unknown] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `g359`: “Haben Tiere im Vergleich zu Menschen oft einen Herzinfakt?”
- `gq2032` [unknown] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `g440`: “Was ist die Heimat von Ursäugern?”
- `gq909` [unknown] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `g175`: “Als was verstehen sich viele griechisch-stämmigen Menschen im Ausland?”
- `gq972` [unknown] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `g185`: “Was ist die Zeitdimension von subklinischen Infektionen?”
- `gq498` [unknown] — positive present only via injected/oracle source — first stage never retrieved it; first-stage nDCG@10 0.0, retriever-oracle 0.0. missing positive `g101`: “In welchem Land hat man das Famicon verkauft?”
