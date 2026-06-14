# v3 candidate build

Status: **OK**

- selected pairs: 24761 (admitted 42683) · passages: 0
- real domains with real pairs: ['faq_real']

## Quota (achieved vs target)

| domain | target | total | real | synthetic | counted | achieved |
|---|--:|--:|--:|--:|--:|:--:|
| admin_real | 6000 | 0 | 0 | 0 | 0 | ❌ |
| cross_lingual_de_en | 2000 | 0 | 0 | 0 | 0 | ❌ |
| faq_real | 6000 | 6000 | 6000 | 0 | 6000 | ✅ |
| german_stress | 2000 | 761 | 0 | 761 | 761 | ❌ |
| legal_adjacency_real_no_eval_overlap | 6000 | 0 | 0 | 0 | 0 | ❌ |
| web | 10000 | 10000 | 10000 | 0 | 10000 | ✅ |
| wiki_non_eval | 8000 | 8000 | 8000 | 0 | 8000 | ✅ |

## Dropped by reason

- blocked_source: 4
- no_data: 0
- missing_fields: 0
- unknown_license: 0
- dedup: 334
- pii: 80
- leakage: 0

## Blocked sources
- admin_real_local (admin_real): license_unverified
- legal_adjacency_real_local (legal_adjacency_real_no_eval_overlap): license_unverified
- mmarco_de (web): license_unverified
- clips_mqa_de (faq_real): license_unverified

## MISSED quotas
- ❌ admin_real: counted 0 < target 6000 (REAL domain)
- ❌ cross_lingual_de_en: counted 0 < target 2000
- ❌ german_stress: counted 761 < target 2000
- ❌ legal_adjacency_real_no_eval_overlap: counted 0 < target 6000 (REAL domain)
