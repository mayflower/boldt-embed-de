# v4 RAG reranker experiment — planned commands

mode=full work_dir=outputs/v4-rag-reranker config=v4-rag-reranker

## build_webfaq_eval
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/build_rag_eval_sets.py --mode webfaq --faq-input /bigdata/johann/src/boldtembed/boldt-embed-de/data/raw/v3/faq_real_local.jsonl --split test --output-dir outputs/v4-rag-reranker/eval/webfaq
```
## build_webfaq_train
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/build_rag_eval_sets.py --mode webfaq --faq-input /bigdata/johann/src/boldtembed/boldt-embed-de/data/raw/v3/faq_real_local.jsonl --split train --output-dir outputs/v4-rag-reranker/train/webfaq
```
## bm25_index_train
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/build_bm25_index.py --corpus outputs/v4-rag-reranker/train/webfaq/corpus.jsonl --output outputs/v4-rag-reranker/first_stage/bm25_train.json
```
## bm25_search_train
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/search_bm25_index.py --index outputs/v4-rag-reranker/first_stage/bm25_train.json --queries outputs/v4-rag-reranker/train/webfaq/queries.jsonl --top-k 20 --output outputs/v4-rag-reranker/first_stage/bm25_train.results.jsonl
```
## build_train_candidate_lists
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/build_rag_candidate_lists.py --queries outputs/v4-rag-reranker/train/webfaq/queries.jsonl --corpus outputs/v4-rag-reranker/train/webfaq/corpus.jsonl --qrels outputs/v4-rag-reranker/train/webfaq/qrels.jsonl --bm25-results outputs/v4-rag-reranker/first_stage/bm25_train.results.jsonl --mode train --output outputs/v4-rag-reranker/candidate_lists/rag_reranker_train_lists.jsonl
```
## teacher_score_train (GPU)
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/score_rag_candidate_lists.py --input outputs/v4-rag-reranker/candidate_lists/rag_reranker_train_lists.jsonl --mode reranker --output outputs/v4-rag-reranker/teacher/rag_train_scored.jsonl --summary outputs/v4-rag-reranker/teacher/rag_teacher_summary.json
```
## train_rag_reranker (GPU)
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/train_rag_reranker_v4.py --candidate-lists outputs/v4-rag-reranker/teacher/rag_train_scored.jsonl --loss mixed_listwise --bf16 --gradient-checkpointing --epochs 1 --batch-size 8 --eval-query-ids outputs/v4-rag-reranker/eval/webfaq/queries.jsonl --output outputs/v4-rag-reranker/checkpoints/boldt-rag-reranker-v4 --run-id v4-rag-reranker
```
## bm25_index_webfaq
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/build_bm25_index.py --corpus outputs/v4-rag-reranker/eval/webfaq/corpus.jsonl --output outputs/v4-rag-reranker/first_stage/bm25_webfaq.json
```
## bm25_search_webfaq
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/search_bm25_index.py --index outputs/v4-rag-reranker/first_stage/bm25_webfaq.json --queries outputs/v4-rag-reranker/eval/webfaq/queries.jsonl --top-k 20 --output outputs/v4-rag-reranker/first_stage/bm25_webfaq.results.jsonl
```
## candidate_lists_webfaq
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/build_rag_candidate_lists.py --queries outputs/v4-rag-reranker/eval/webfaq/queries.jsonl --corpus outputs/v4-rag-reranker/eval/webfaq/corpus.jsonl --qrels outputs/v4-rag-reranker/eval/webfaq/qrels.jsonl --bm25-results outputs/v4-rag-reranker/first_stage/bm25_webfaq.results.jsonl --mode eval --output outputs/v4-rag-reranker/candidate_lists/eval_webfaq_lists.jsonl
```
## lift_webfaq (GPU)
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/eval_rag_reranker_lift.py --reranker outputs/v4-rag-reranker/checkpoints/boldt-rag-reranker-v4 --candidate-lists outputs/v4-rag-reranker/candidate_lists/eval_webfaq_lists.jsonl --eval-set webfaq --output outputs/v4-rag-reranker/eval/reranker_lift_webfaq.json
```
## bm25_index_germanquad
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/build_bm25_index.py --corpus /bigdata/johann/src/boldtembed/boldt-embed-de/data/processed/eval/germanquad_corpus.jsonl --output outputs/v4-rag-reranker/first_stage/bm25_germanquad.json
```
## bm25_search_germanquad
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/search_bm25_index.py --index outputs/v4-rag-reranker/first_stage/bm25_germanquad.json --queries /bigdata/johann/src/boldtembed/boldt-embed-de/data/processed/eval/germanquad_queries.jsonl --top-k 20 --output outputs/v4-rag-reranker/first_stage/bm25_germanquad.results.jsonl
```
## candidate_lists_germanquad
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/build_rag_candidate_lists.py --queries /bigdata/johann/src/boldtembed/boldt-embed-de/data/processed/eval/germanquad_queries.jsonl --corpus /bigdata/johann/src/boldtembed/boldt-embed-de/data/processed/eval/germanquad_corpus.jsonl --qrels /bigdata/johann/src/boldtembed/boldt-embed-de/data/processed/eval/germanquad_qrels.jsonl --bm25-results outputs/v4-rag-reranker/first_stage/bm25_germanquad.results.jsonl --mode eval --output outputs/v4-rag-reranker/candidate_lists/eval_germanquad_lists.jsonl
```
## lift_germanquad (GPU)
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/eval_rag_reranker_lift.py --reranker outputs/v4-rag-reranker/checkpoints/boldt-rag-reranker-v4 --candidate-lists outputs/v4-rag-reranker/candidate_lists/eval_germanquad_lists.jsonl --eval-set germanquad --output outputs/v4-rag-reranker/eval/reranker_lift_germanquad.json
```
## bm25_index_dt_test
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/build_bm25_index.py --corpus /bigdata/johann/src/boldtembed/boldt-embed-de/data/processed/eval/dt_test_corpus.jsonl --output outputs/v4-rag-reranker/first_stage/bm25_dt_test.json
```
## bm25_search_dt_test
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/search_bm25_index.py --index outputs/v4-rag-reranker/first_stage/bm25_dt_test.json --queries /bigdata/johann/src/boldtembed/boldt-embed-de/data/processed/eval/dt_test_queries.jsonl --top-k 20 --output outputs/v4-rag-reranker/first_stage/bm25_dt_test.results.jsonl
```
## candidate_lists_dt_test
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/build_rag_candidate_lists.py --queries /bigdata/johann/src/boldtembed/boldt-embed-de/data/processed/eval/dt_test_queries.jsonl --corpus /bigdata/johann/src/boldtembed/boldt-embed-de/data/processed/eval/dt_test_corpus.jsonl --qrels /bigdata/johann/src/boldtembed/boldt-embed-de/data/processed/eval/dt_test_qrels.jsonl --bm25-results outputs/v4-rag-reranker/first_stage/bm25_dt_test.results.jsonl --mode eval --output outputs/v4-rag-reranker/candidate_lists/eval_dt_test_lists.jsonl
```
## lift_dt_test (GPU)
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/eval_rag_reranker_lift.py --reranker outputs/v4-rag-reranker/checkpoints/boldt-rag-reranker-v4 --candidate-lists outputs/v4-rag-reranker/candidate_lists/eval_dt_test_lists.jsonl --eval-set dt_test --output outputs/v4-rag-reranker/eval/reranker_lift_dt_test.json
```
## promotion_gate [GATE]
```
/home/johann/anaconda3/envs/boldtembed/bin/python /bigdata/johann/src/boldtembed/boldt-embed-de/scripts/check_rag_reranker_promotion_gate.py --eval-dir outputs/v4-rag-reranker/eval --output outputs/v4-rag-reranker/eval/rag_reranker_gate.json --markdown outputs/v4-rag-reranker/eval/rag_reranker_gate.md
```
