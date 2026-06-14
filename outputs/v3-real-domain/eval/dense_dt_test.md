# Baseline benchmark report

commit: `f4c8b029182e7ee9659dafb04901558e44092996` · torch: 2.6.0+cu124 · sentence-transformers: 5.5.1

| model | task | nDCG@10 | MRR@10 | Recall@10 | Recall@100 | MAP@10 |
|---|---|---:|---:|---:|---:|---:|
| Boldt/Boldt-DC-350M | dt_test | 0.22260695311181372 | 0.19505753968253967 | 0.312 | 0.546 | 0.19505753968253967 |
| outputs/v3-real-domain/checkpoints/boldt-modern-causal-v3 | dt_test | 0.9700865905684374 | 0.9646095238095238 | 0.987 | 0.998 | 0.9646095238095238 |
| intfloat/multilingual-e5-base | dt_test | 0.9935873898144528 | 0.9921583333333333 | 0.998 | 1.0 | 0.9921583333333333 |
