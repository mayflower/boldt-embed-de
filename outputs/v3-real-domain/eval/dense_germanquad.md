# Baseline benchmark report

commit: `f4c8b029182e7ee9659dafb04901558e44092996` · torch: 2.6.0+cu124 · sentence-transformers: 5.5.1

| model | task | nDCG@10 | MRR@10 | Recall@10 | Recall@100 | MAP@10 |
|---|---|---:|---:|---:|---:|---:|
| Boldt/Boldt-DC-350M | germanquad | 0.2879650789364867 | 0.25563492063492066 | 0.39066666666666666 | 0.5293333333333333 | 0.25563492063492066 |
| outputs/v3-real-domain/checkpoints/boldt-modern-causal-v3 | germanquad | 0.8851625750107249 | 0.8557685185185185 | 0.974 | 0.9966666666666667 | 0.8557685185185185 |
| intfloat/multilingual-e5-base | germanquad | 0.9388648472161265 | 0.9236515873015874 | 0.984 | 0.998 | 0.9236515873015874 |
