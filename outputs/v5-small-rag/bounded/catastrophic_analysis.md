# v5 catastrophic GermanQuAD drop analysis

- catastrophic queries: **185** of 1500
- avg first-stage gap: 5.6291; avg reranker gap: 0.6474
- **policy-fixable (any bounded policy): 185 (100.0%)**; data/model-fixable (none): 0

## Counts by error type

| error_type | count |
|---|--:|
| positive_demoted_from_top1_or_top3 | 6 |
| lexical_exact_positive_demoted | 8 |
| reranker_promotes_longer_but_wrong_doc | 26 |
| query_style_mismatch | 135 |
| insufficient_first_stage_features | 9 |
| unknown | 1 |

## Fixed by policy (of the catastrophic drops)

| policy | fixes |
|---|--:|
| top1_lock | 172 |
| top3_lock | 183 |
| bounded_downshift_D1 | 15 |
| blend_alpha0.85 | 181 |
| margin_override_M3 | 167 |

## Top 20 catastrophic examples

| query_id | fs→rr nDCG | Δ | pos init→final rank | error_type | fixed_by |
|---|---|--:|---|---|---|
| gq773 | 1.0→0.0 | -1.0 | 0→11 | lexical_exact_positive_demoted | top1_lock,top3_lock,blend_alpha0.85,margin_override_M3 |
| gq1703 | 1.0→0.0 | -1.0 | 0→14 | lexical_exact_positive_demoted | top1_lock,top3_lock,blend_alpha0.85 |
| gq1099 | 1.0→0.0 | -1.0 | 0→16 | lexical_exact_positive_demoted | top1_lock,top3_lock,blend_alpha0.85,margin_override_M3 |
| gq1261 | 1.0→0.0 | -1.0 | 0→11 | lexical_exact_positive_demoted | top1_lock,top3_lock,blend_alpha0.85,margin_override_M3 |
| gq1001 | 1.0→0.0 | -1.0 | 0→16 | lexical_exact_positive_demoted | top1_lock,top3_lock,blend_alpha0.85,margin_override_M3 |
| gq919 | 1.0→0.0 | -1.0 | 0→12 | lexical_exact_positive_demoted | top1_lock,top3_lock,blend_alpha0.85,margin_override_M3 |
| gq928 | 1.0→0.0 | -1.0 | 0→10 | lexical_exact_positive_demoted | top1_lock,top3_lock,blend_alpha0.85,margin_override_M3 |
| gq256 | 1.0→0.0 | -1.0 | 0→13 | lexical_exact_positive_demoted | top1_lock,top3_lock,blend_alpha0.85,margin_override_M3 |
| gq1226 | 1.0→0.2891 | -0.7109 | 0→9 | query_style_mismatch | top1_lock,top3_lock,blend_alpha0.85,margin_override_M3 |
| gq857 | 1.0→0.2891 | -0.7109 | 0→9 | query_style_mismatch | top1_lock,top3_lock,blend_alpha0.85,margin_override_M3 |
| gq915 | 1.0→0.2891 | -0.7109 | 0→9 | query_style_mismatch | top1_lock,top3_lock,blend_alpha0.85,margin_override_M3 |
| gq916 | 1.0→0.2891 | -0.7109 | 0→9 | query_style_mismatch | top1_lock,top3_lock,blend_alpha0.85,margin_override_M3 |
| gq379 | 1.0→0.2891 | -0.7109 | 0→9 | query_style_mismatch | top1_lock,top3_lock,blend_alpha0.85,margin_override_M3 |
| gq622 | 1.0→0.301 | -0.699 | 0→8 | query_style_mismatch | top1_lock,top3_lock,blend_alpha0.85,margin_override_M3 |
| gq356 | 1.0→0.301 | -0.699 | 0→8 | query_style_mismatch | top1_lock,top3_lock,blend_alpha0.85,margin_override_M3 |
| gq936 | 1.0→0.301 | -0.699 | 0→8 | query_style_mismatch | top1_lock,top3_lock,blend_alpha0.85,margin_override_M3 |
| gq459 | 1.0→0.301 | -0.699 | 0→8 | query_style_mismatch | top1_lock,top3_lock,blend_alpha0.85,margin_override_M3 |
| gq1013 | 1.0→0.301 | -0.699 | 0→8 | query_style_mismatch | top1_lock,top3_lock,blend_alpha0.85 |
| gq666 | 1.0→0.3155 | -0.6845 | 0→7 | query_style_mismatch | top1_lock,top3_lock,blend_alpha0.85,margin_override_M3 |
| gq878 | 1.0→0.3155 | -0.6845 | 0→7 | query_style_mismatch | top1_lock,top3_lock,blend_alpha0.85,margin_override_M3 |
