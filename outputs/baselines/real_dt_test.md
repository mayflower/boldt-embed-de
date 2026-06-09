# Baseline benchmark report

commit: `dfeab5cc77c3c325a92ce19bc26e073c9a10aec2` · torch: 2.6.0+cu124 · sentence-transformers: 5.5.1

| model | task | nDCG@10 | MRR@10 | Recall@10 | Recall@100 | MAP@10 |
|---|---|---:|---:|---:|---:|---:|
| Boldt/Boldt-DC-350M | dt_test | 0.22260695311181372 | 0.19505753968253967 | 0.312 | 0.546 | 0.19505753968253967 |
| outputs/checkpoints/boldt-modern-de | dt_test | 0.9501286162324378 | 0.9395623015873016 | 0.982 | 0.996 | 0.9395623015873016 |
| intfloat/multilingual-e5-base | dt_test | 0.9935873898144528 | 0.9921583333333333 | 0.998 | 1.0 | 0.9921583333333333 |
