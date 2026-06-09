# Experiments

5 run card(s).

| run_id | type | model | dataset | key metrics | commit | date |
|---|---|---|---|---|---|---|
| real-teacher-cache | teacher_cache | Qwen/Qwen3-Embedding-8B + Qwen/Qwen3-Reranker-8B | data/processed/candidates.jsonl | — | `dfeab5cc` | 2026-06-09T12:49:51.330303+00:00 |
| real-train-embedder | train_embedder | Boldt/Boldt-DC-350M | outputs/teacher-cache/cache_train_pos.jsonl | — | `dfeab5cc` | 2026-06-09T12:57:45.244323+00:00 |
| real-eval-germanquad | eval | — | germanquad | ndcg@10=0.2879650789364867, mrr@10=0.25563492063492066, recall@100=0.5293333333333333 | `dfeab5cc` | 2026-06-09T16:09:21.249661+00:00 |
| real-eval-dt_test | eval | — | dt_test | ndcg@10=0.22260695311181372, mrr@10=0.19505753968253967, recall@100=0.546 | `dfeab5cc` | 2026-06-09T16:10:51.773275+00:00 |
| real-eval-gerdalir | eval | — | gerdalir | ndcg@10=0.0021202301571267164, mrr@10=0.002416666666666667, recall@100=0.02033333333333333 | `dfeab5cc` | 2026-06-09T16:15:45.371624+00:00 |
