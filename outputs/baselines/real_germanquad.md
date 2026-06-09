# Baseline benchmark report

commit: `dfeab5cc77c3c325a92ce19bc26e073c9a10aec2` · torch: 2.6.0+cu124 · sentence-transformers: 5.5.1

| model | task | nDCG@10 | MRR@10 | Recall@10 | Recall@100 | MAP@10 |
|---|---|---:|---:|---:|---:|---:|
| Boldt/Boldt-DC-350M | germanquad | 0.2879650789364867 | 0.25563492063492066 | 0.39066666666666666 | 0.5293333333333333 | 0.25563492063492066 |
| outputs/checkpoints/boldt-modern-de | germanquad | 0.8831334574978987 | 0.8525558201058201 | 0.9753333333333334 | 0.9966666666666667 | 0.8525558201058201 |
| intfloat/multilingual-e5-base | germanquad | 0.9388648472161265 | 0.9236515873015874 | 0.984 | 0.998 | 0.9236515873015874 |
