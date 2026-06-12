# Baseline benchmark report

commit: `1a05590191d9573c84054069f4029bae34bbff22` · torch: 2.6.0+cu124 · sentence-transformers: 5.5.1

| model | task | nDCG@10 | MRR@10 | Recall@10 | Recall@100 | MAP@10 |
|---|---|---:|---:|---:|---:|---:|
| Boldt/Boldt-DC-350M | dt_test | 0.22260695311181372 | 0.19505753968253967 | 0.312 | 0.546 | 0.19505753968253967 |
| outputs/v2-generalization/checkpoints/boldt-modern-causal-v2 | dt_test | 0.9442777883635152 | 0.9350559523809524 | 0.973 | 0.997 | 0.9350559523809524 |
| intfloat/multilingual-e5-base | dt_test | 0.9935873898144528 | 0.9921583333333333 | 0.998 | 1.0 | 0.9921583333333333 |
