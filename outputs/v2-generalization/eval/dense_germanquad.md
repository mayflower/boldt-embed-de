# Baseline benchmark report

commit: `1a05590191d9573c84054069f4029bae34bbff22` · torch: 2.6.0+cu124 · sentence-transformers: 5.5.1

| model | task | nDCG@10 | MRR@10 | Recall@10 | Recall@100 | MAP@10 |
|---|---|---:|---:|---:|---:|---:|
| Boldt/Boldt-DC-350M | germanquad | 0.2879650789364867 | 0.25563492063492066 | 0.39066666666666666 | 0.5293333333333333 | 0.25563492063492066 |
| outputs/v2-generalization/checkpoints/boldt-modern-causal-v2 | germanquad | 0.8857504605906706 | 0.8560637566137566 | 0.9753333333333334 | 0.998 | 0.8560637566137566 |
| intfloat/multilingual-e5-base | germanquad | 0.9388648472161265 | 0.9236515873015874 | 0.984 | 0.998 | 0.9236515873015874 |
