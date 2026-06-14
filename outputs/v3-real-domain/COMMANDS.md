# v3 real-domain experiment — planned commands

mode=dry-run target_count=1000 work_dir=/bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain

## acquire_sources
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/acquire_v3_sources.py --manifest /bigdata/johann/src/boldtembed/boldt-embed-de/configs/data_sources_v3.json --output-dir data/raw/v3 --mode dry-run
```
## build_leakage_index
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/build_leakage_index.py --eval-corpus /bigdata/johann/src/boldtembed/boldt-embed-de/data/processed/eval/germanquad_corpus.jsonl /bigdata/johann/src/boldtembed/boldt-embed-de/data/processed/eval/germanquad_queries.jsonl /bigdata/johann/src/boldtembed/boldt-embed-de/data/processed/eval/dt_test_corpus.jsonl /bigdata/johann/src/boldtembed/boldt-embed-de/data/processed/eval/dt_test_queries.jsonl --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/leakage/eval_index.json
```
## build_candidates
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/build_v3_candidates.py --manifest /bigdata/johann/src/boldtembed/boldt-embed-de/configs/data_sources_v3.json --config /bigdata/johann/src/boldtembed/boldt-embed-de/configs/experiments/v3_real_domain_generalization.json --raw-dir data/raw/v3 --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/candidates_v3.jsonl --target-count 1000 --leakage-index /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/leakage/eval_index.json --pii-scan --fail-on-unknown-license --fail-on-domain-quota-miss
```
## full_leakage_scan
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/run_full_leakage_scan.py --candidates /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/candidates_v3.jsonl --eval-corpus /bigdata/johann/src/boldtembed/boldt-embed-de/data/processed/eval/germanquad_corpus.jsonl /bigdata/johann/src/boldtembed/boldt-embed-de/data/processed/eval/germanquad_queries.jsonl /bigdata/johann/src/boldtembed/boldt-embed-de/data/processed/eval/dt_test_corpus.jsonl /bigdata/johann/src/boldtembed/boldt-embed-de/data/processed/eval/dt_test_queries.jsonl --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/leakage/leakage_report.json --hits-output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/leakage/leakage_hits.jsonl --drop-hits /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/candidates_v3.clean.jsonl
```
## build_bm25_index
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/build_bm25_index.py --corpus /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/candidates_v3.clean.jsonl --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/bm25_v3.json
```
## teacher_cache (GPU)
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/build_teacher_cache.py --input /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/candidates_v3.clean.jsonl --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/teacher-cache/qwen3_v3.jsonl --mode both --shard-size 5000 --max-length 512 --resume
```
## summarize_cache
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/summarize_teacher_cache.py --input /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/teacher-cache/qwen3_v3.manifest.json --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/teacher-cache/qwen3_v3.summary.json --fail-on-unknown-license --fail-on-disallowed-training-source
```
## calibrate_thresholds
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/calibrate_teacher_thresholds.py --teacher-cache /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/teacher-cache/qwen3_v3.manifest.json --config /bigdata/johann/src/boldtembed/boldt-embed-de/configs/experiments/v3_real_domain_generalization.json --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/teacher-cache/qwen3_v3.calibration.json --markdown /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/teacher-cache/qwen3_v3.calibration.md --embedder-output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/teacher-cache/qwen3_v3.filtered_embedder.jsonl --reranker-output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/teacher-cache/qwen3_v3.filtered_reranker.jsonl
```
## domain_quality_gate [GATE]
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/analyze_domain_quality.py --candidates /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/candidates_v3.clean.jsonl --teacher-cache /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/teacher-cache/qwen3_v3.manifest.json --config /bigdata/johann/src/boldtembed/boldt-embed-de/configs/experiments/v3_real_domain_generalization.json --source-manifest /bigdata/johann/src/boldtembed/boldt-embed-de/configs/data_sources_v3.json --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/domain_quality.json --markdown /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/domain_quality.md
```
## mine_hard_negatives
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/mine_hard_negatives_2026.py --candidates /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/teacher-cache/qwen3_v3.filtered_embedder.jsonl --teacher-cache /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/teacher-cache/qwen3_v3.jsonl --bm25-index /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/bm25_v3.json --output /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain/hardneg_v3.jsonl --negatives-per-query 8 --require-full-corpus
```
## release_gate [GATE]
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/validate_release_2026.py --require-v3-artifacts --results-dir /bigdata/johann/src/boldtembed/boldt-embed-de/outputs/v3-real-domain
```
