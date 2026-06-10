# Experiments

9 run card(s).

| run_id | type | model | dataset | key metrics | commit | date |
|---|---|---|---|---|---|---|
| real-teacher-cache | teacher_cache | Qwen/Qwen3-Embedding-8B + Qwen/Qwen3-Reranker-8B | data/processed/candidates.jsonl | — | `dfeab5cc` | 2026-06-09T12:49:51.330303+00:00 |
| real-train-embedder | train_embedder | Boldt/Boldt-DC-350M | outputs/teacher-cache/cache_train_pos.jsonl | — | `dfeab5cc` | 2026-06-09T12:57:45.244323+00:00 |
| real-eval-germanquad | eval | — | germanquad | ndcg@10=0.2879650789364867, mrr@10=0.25563492063492066, recall@100=0.5293333333333333 | `dfeab5cc` | 2026-06-09T16:09:21.249661+00:00 |
| real-eval-dt_test | eval | — | dt_test | ndcg@10=0.22260695311181372, mrr@10=0.19505753968253967, recall@100=0.546 | `dfeab5cc` | 2026-06-09T16:10:51.773275+00:00 |
| real-eval-gerdalir | eval | — | gerdalir | ndcg@10=0.0021202301571267164, mrr@10=0.002416666666666667, recall@100=0.02033333333333333 | `dfeab5cc` | 2026-06-09T16:15:45.371624+00:00 |
| real-rr-teacher-cache | teacher_cache | Qwen/Qwen3-Embedding-8B + Qwen/Qwen3-Reranker-8B | data/processed/teacher_rr_input.jsonl | — | `9f34fb98` | 2026-06-10T06:54:26.879087+00:00 |
| real-train-reranker | train_reranker | Boldt/Boldt-DC-350M | data/processed/reranker_train.jsonl | final_loss=0.02311387099325657 | `9f34fb98` | 2026-06-10T07:42:17.637558+00:00 |
| real-lift-germanquad | eval | outputs/checkpoints/boldt-reranker-modern | data/processed/eval/germanquad_shortlist.jsonl | — | `9f34fb98` | 2026-06-10T07:47:53.062089+00:00 |
| real-lift-dt_test | eval | outputs/checkpoints/boldt-reranker-modern | data/processed/eval/dt_test_shortlist.jsonl | — | `9f34fb98` | 2026-06-10T07:51:18.614613+00:00 |
