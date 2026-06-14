# v3 domain-quality report

Status: **FAIL**

- raw candidates: 24761 · cache rows: 24761 · accepted positives: 22736
- unknown-license rows: 0 · disallowed-source rows: 0
- effective web+wiki share: 0.7802 (max 0.65)
- can claim legal transfer from data: **False**

## Per-domain

| domain | raw | accepted | accept% | synth share | eff share | med rerank | suspicious |
|---|--:|--:|--:|--:|--:|--:|--:|
| faq_real | 6000 | 4248 | 0.708 | 0.0 | 0.1868 | 5.9375 | 1752 |
| german_stress | 761 | 750 | 0.9855 | 1.0 | 0.033 | 6.375 | 11 |
| web | 10000 | 9879 | 0.9879 | 0.0 | 0.4345 | 7.9375 | 121 |
| wiki_non_eval | 8000 | 7859 | 0.9824 | 0.0 | 0.3457 | 6.6875 | 141 |

## Gates

**FAILING gates:**
- ❌ real_domain_min_accepted [faq_real]: accepted=4248 (min 5000)
- ❌ real_domain_min_raw [admin_real]: raw=0 (min 5000)
- ❌ real_domain_min_accepted [admin_real]: accepted=0 (min 5000)
- ❌ real_domain_acceptance_rate [admin_real]: acceptance_rate=0.0 (min 0.35)
- ❌ real_domain_min_raw [legal_adjacency_real_no_eval_overlap]: raw=0 (min 5000)
- ❌ real_domain_min_accepted [legal_adjacency_real_no_eval_overlap]: accepted=0 (min 5000)
- ❌ real_domain_acceptance_rate [legal_adjacency_real_no_eval_overlap]: acceptance_rate=0.0 (min 0.35)
- ❌ effective_web_wiki_share: web+wiki share=0.7802 (max 0.65)
