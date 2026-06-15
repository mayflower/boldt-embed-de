# bounded rerank policy fit — **confidence_conditional**

Fit on DEV ONLY: `outputs/v5-small-rag/eval/webfaq_dev_scored_conservative.jsonl` (660 lists). Guardrails not used.

- selected params: `{"fs_gap_high": 10.314034, "k": 1}`
- observable safety (high-conf dev top-1 keep): {'hc_gap_threshold': 6.9251, 'hc_lists': 266, 'safety_top1_keep_min': 0.9, 'selected_safe': True}
- selected a safe policy: True
- dev nDCG@10: 0.756512 (delta +0.143342), dev catastrophic: 0.009091, hc_top1_keep: 0.9812

## Top dev candidates (highest high-conf top-1 keep, then highest nDCG)

| policy | params | dev nDCG@10 | dev delta | hc_top1_keep | catastrophic |
|---|---|--:|--:|--:|--:|
| confidence_conditional | {"fs_gap_high": 5.017926, "k": 1} | 0.754636 | +0.141466 | 1.0 | 0.007576 |
| confidence_conditional | {"fs_gap_high": 5.017926, "k": 3} | 0.750682 | +0.137512 | 1.0 | 0.007576 |
| margin_override | {"margin": 2.0} | 0.728075 | +0.114905 | 1.0 | 0.001515 |
| margin_override | {"margin": 3.0} | 0.719783 | +0.106613 | 1.0 | 0.001515 |
| margin_override | {"margin": 5.0} | 0.714533 | +0.101363 | 1.0 | 0.0 |
| top1_lock | {} | 0.703471 | +0.090301 | 1.0 | 0.0 |
| topk_lock | {"k": 1} | 0.703471 | +0.090301 | 1.0 | 0.0 |
| topk_lock | {"k": 2} | 0.68765 | +0.07448 | 1.0 | 0.0 |
| topk_lock | {"k": 3} | 0.679817 | +0.066647 | 1.0 | 0.0 |
| topk_lock | {"k": 5} | 0.662322 | +0.049152 | 1.0 | 0.0 |
| blend | {"alpha": 0.7} | 0.631004 | +0.017834 | 1.0 | 0.00303 |
| blend | {"alpha": 0.9} | 0.618731 | +0.005561 | 1.0 | 0.001515 |
| identity | {} | 0.61317 | +0.0 | 1.0 | 0.0 |
| topk_lock | {"k": 10} | 0.61317 | +0.0 | 1.0 | 0.0 |
| bounded_upshift | {"U": 2, "margin": 2.0} | 0.67762 | +0.06445 | 0.9925 | 0.006061 |
