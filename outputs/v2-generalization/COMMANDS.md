# v2 experiment — planned commands

mode=dry-run target_count=50000 work_dir=/bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization

## build_candidates
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/build_v2_candidates.py --manifest /bigdata/johann/src/boldtembed/boldt-embed-de/configs/data_sources_v2.json --source-jsonl data/raw/v2/*.jsonl --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/candidates_v2.jsonl --domain-config /bigdata/johann/src/boldtembed/boldt-embed-de/configs/experiments/v2_generalization.json --target-count 50000 --dedup --pii-scan
```
## generate_synthetic
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/generate_synthetic_queries.py --passages /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/passages.jsonl --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/synthetic_v2.jsonl --families germanquad web faq admin --queries-per-passage 4
```
## teacher_cache (GPU)
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/build_teacher_cache.py --input /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/candidates_v2.jsonl --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/teacher-cache/qwen3_v2.jsonl --mode both --shard-size 5000 --max-length 512
```
## summarize_filter_cache
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/summarize_teacher_cache.py --input /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/teacher-cache/qwen3_v2.manifest.json --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/teacher-cache/qwen3_v2.summary.json --filter-output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/teacher-cache/qwen3_v2.filtered.jsonl --reranker-threshold 2.0
```
## mine_hard_negatives
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/mine_hard_negatives_2026.py --candidates /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/candidates_v2.jsonl --teacher-cache /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/teacher-cache/qwen3_v2.filtered.jsonl --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/hardneg_v2.jsonl --negatives-per-query 8
```
## reranker_candidate_lists
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/build_reranker_candidates_v2.py --candidates /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/candidates_v2.jsonl --teacher-cache /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/teacher-cache/qwen3_v2.filtered.jsonl --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/reranker_train_v2.jsonl
```
## train_causal (GPU)
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/train_modern_embedder.py --teacher-cache /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/teacher-cache/qwen3_v2.filtered.jsonl --hard-negatives /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/hardneg_v2.jsonl --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/checkpoints/boldt-modern-causal-v2 --bf16 --gradient-checkpointing --run-id v2-causal
```
## prepare_bi_mntp (GPU)
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/prepare_bidirectional_student.py --texts /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/mntp_texts.jsonl --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/checkpoints/boldt-bi-mntp-v2 --bf16 --gradient-checkpointing --run-id v2-mntp
```
## train_bi_mntp (GPU)
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/train_modern_embedder.py --base-model /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/checkpoints/boldt-bi-mntp-v2 --bidirectional true --teacher-cache /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/teacher-cache/qwen3_v2.filtered.jsonl --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/checkpoints/boldt-modern-bi-mntp-v2 --bf16 --gradient-checkpointing --run-id v2-bi-mntp
```
## train_reranker (GPU)
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/train_modern_reranker.py --candidate-lists /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/reranker_train_v2.jsonl --loss mixed --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/checkpoints/boldt-reranker-modern-v2 --bf16 --run-id v2-reranker
```
## eval_dense_germanquad (GPU)
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/run_baseline_benchmarks.py --models data/processed/v2_eval_models.json --mode local --task-name germanquad --eval-corpus data/processed/eval/germanquad_corpus.jsonl --eval-queries data/processed/eval/germanquad_queries.jsonl --qrels data/processed/eval/germanquad_qrels.jsonl --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/eval/dense_germanquad.json --run-id v2-eval-germanquad
```
## eval_dense_dt_test (GPU)
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/run_baseline_benchmarks.py --models data/processed/v2_eval_models.json --mode local --task-name dt_test --eval-corpus data/processed/eval/dt_test_corpus.jsonl --eval-queries data/processed/eval/dt_test_queries.jsonl --qrels data/processed/eval/dt_test_qrels.jsonl --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/eval/dense_dt_test.json --run-id v2-eval-dt_test
```
## eval_dense_gerdalir (GPU)
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/run_baseline_benchmarks.py --models data/processed/v2_eval_models.json --mode local --task-name gerdalir --eval-corpus data/processed/eval/gerdalir_corpus.jsonl --eval-queries data/processed/eval/gerdalir_queries.jsonl --qrels data/processed/eval/gerdalir_qrels.jsonl --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/eval/dense_gerdalir.json --run-id v2-eval-gerdalir
```
## summarize_results
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/summarize_v2_results.py --v1-dir outputs --v2-dir /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization --config /bigdata/johann/src/boldtembed/boldt-embed-de/configs/experiments/v2_generalization.json --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/V2_RESULTS.md --json-output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v2-generalization/V2_RESULTS.json
```
