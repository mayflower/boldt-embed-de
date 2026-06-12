# Baseline benchmark report

commit: `1a05590191d9573c84054069f4029bae34bbff22` · torch: 2.6.0+cu124 · sentence-transformers: 5.5.1

| model | task | nDCG@10 | MRR@10 | Recall@10 | Recall@100 | MAP@10 |
|---|---|---:|---:|---:|---:|---:|
| Boldt/Boldt-DC-350M | gerdalir | 0.0029451816598497025 | 0.0028409960791029943 | 0.004563783989973299 | 0.02027624029830994 | 0.0022799458184450828 |
| outputs/v2-generalization/checkpoints/boldt-modern-causal-v2 | gerdalir | 0.10958224482523524 | 0.0988917173321293 | 0.16339845240041415 | 0.34169060463812795 | 0.08966149989231156 |
| intfloat/multilingual-e5-base | gerdalir | 0.1530158905028875 | 0.1403068731170742 | 0.2178376810917272 | 0.4041218072973835 | 0.12884123297203645 |
