# v4 German RAG reranker — results

## 1. Executive verdict: **mixed**  (promotion gate: fail)

Keep the reranker DISABLED for production; experimental only.

## 2. Training data

- WebFAQ train candidate lists: 7415
- teacher-scored lists: 7415
- gold positives / hard negatives / uncertain: 7415 / 125567 / 14600
- excluded eval splits (never trained): webfaq_heldout, germanquad, dt_test

## 3. Reranker lift (nDCG@10 over FIXED candidates)

| eval set | first stage | reranked | delta | diagnostic |
|---|--:|--:|--:|:--:|
| dt_test | 0.9774 | 0.9767 | -0.0007 |  |
| germanquad | 0.9058 | 0.8347 | -0.0711 |  |
| webfaq | 0.5945 | 0.8852 | +0.2907 |  |

## 4. First-stage recall

| eval set | positive_in_top_10 | oracle_ndcg@10 |
|---|--:|--:|
| dt_test | 0.992 | 1.0 |
| germanquad | 0.9613 | 1.0 |
| webfaq | 0.6478 | 1.0 |

## 5. Teacher / student (nDCG@10)

| eval set | Boldt v4 | Qwen teacher |
|---|--:|--:|
| dt_test | 0.9767 | n/a |
| germanquad | 0.8347 | n/a |
| webfaq | 0.8852 | n/a |

## 6. Failure cases

- reranker hurts (delta < 0): ['dt_test', 'germanquad']
- score-calibration suspect (positive_in_top_10 dropped): none

## 7. Decision

Keep the reranker DISABLED for production; experimental only.
