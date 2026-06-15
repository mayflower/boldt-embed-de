# v5 small-RAG (prompt-4) — reproducible commands for the REAL run

Executed 2026-06-14/15 on RTX A6000. Heavy artifacts (weights, teacher-scored JSONL, candidate
lists, raw data) are git-ignored; this file + the small JSON/MD summaries are the record. Re-run
top-to-bottom to reproduce.

```bash
# 1. Acquire REAL multi-domain German data, leakage-filtered vs dt_test + GermanQuAD.
#    Sources: WebFAQ (faq_real) + deutsche-telekom/wikipedia-22-12-de-dpr TRAIN split
#    (qa_passage_non_eval / german_stress / long_doc_chunks). web_nonfaq + local_rag omitted (no real source).
HF_HUB_OFFLINE=1 HF_HOME=/bigdata/johann/hf-cache python scripts/acquire_v5_real.py
#    -> data/raw/v5/*.jsonl  (6,500 rows; faq share 0.3077; 0 leakage contexts — DPR train disjoint)

# 2. Prep corpus/queries/qrels (queries carry domain).
python scripts/prep_v5_candidate_lists.py --stage prep
#    -> outputs/v5-small-rag/train/{corpus,queries,qrels}.jsonl + domain_map.json  (6,500 q / 4,883 docs)

# 3. BM25 first stage.
python scripts/build_bm25_index.py  --corpus outputs/v5-small-rag/train/corpus.jsonl  --output outputs/v5-small-rag/train/bm25.json
python scripts/search_bm25_index.py --index outputs/v5-small-rag/train/bm25.json --queries outputs/v5-small-rag/train/queries.jsonl --top-k 20 --output outputs/v5-small-rag/train/bm25.results.jsonl

# 4. Build fixed candidate lists (train mode, inject positive) + attach domain.
python scripts/build_rag_candidate_lists.py --queries outputs/v5-small-rag/train/queries.jsonl --corpus outputs/v5-small-rag/train/corpus.jsonl --qrels outputs/v5-small-rag/train/qrels.jsonl --bm25-results outputs/v5-small-rag/train/bm25.results.jsonl --mode train --top-k 20 --output outputs/v5-small-rag/candidate_lists/rag_train_lists.jsonl
python scripts/prep_v5_candidate_lists.py --stage attach --candidate-lists outputs/v5-small-rag/candidate_lists/rag_train_lists.jsonl --domain-map outputs/v5-small-rag/train/domain_map.json
#    -> 5,660 lists (BM25 recall 0.8708); faq share 0.217

# 5. Teacher-score the candidate lists with Qwen3-Reranker-8B (GPU, ~2h for 113,145 pairs).
CUDA_VISIBLE_DEVICES=0 HF_HOME=/bigdata/johann/hf-cache HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/score_rag_candidate_lists.py --input outputs/v5-small-rag/candidate_lists/rag_train_lists.jsonl --teacher-config configs/teacher_models.json --mode reranker --output outputs/v5-small-rag/teacher/rag_train_scored.jsonl --summary outputs/v5-small-rag/teacher/rag_teacher_summary.json

# 6. Train the v5 reranker (GPU, listwise-KL primary).
CUDA_VISIBLE_DEVICES=0 HF_HOME=/bigdata/johann/hf-cache HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/train_v5_rag_reranker.py --config configs/experiments/v5_small_rag.json --candidate-lists outputs/v5-small-rag/teacher/rag_train_scored.jsonl --model-base Boldt/Boldt-DC-350M --output outputs/v5-small-rag/checkpoints/boldt-rag-reranker-v5 --loss listwise_kl+pairwise+pointwise_confident --max-faq-share 0.35 --bf16 --gradient-checkpointing --run-id v5-reranker-boldt

# 7. Score fixed eval lists with the v5 reranker (model loaded once), then run the hardness-aware gate.
#    WebFAQ eval is the v4 fixed lists MINUS 216 queries that overlapped v5 training (leakage-filtered).
python scripts/score_eval_lists_v5.py --reranker outputs/v5-small-rag/checkpoints/boldt-rag-reranker-v5 --lists webfaq=outputs/v5-small-rag/eval/webfaq_heldout_lists.jsonl germanquad=outputs/v4-rag-reranker/candidate_lists/eval_germanquad_lists.jsonl dt_test=outputs/v4-rag-reranker/candidate_lists/eval_dt_test_lists.jsonl --out-dir outputs/v5-small-rag/eval/v5_scored
python scripts/eval_v5_rag_lift.py --primary webfaq=outputs/v5-small-rag/eval/v5_scored/webfaq_scored.jsonl --guardrail germanquad=outputs/v5-small-rag/eval/v5_scored/germanquad_scored.jsonl dt_test=outputs/v5-small-rag/eval/v5_scored/dt_test_scored.jsonl --report outputs/v5-small-rag/eval/v5_rag_lift_gate.json

# 8. Summarize.
python scripts/summarize_v5_results.py   # -> V5_RESULTS.{md,json}
```

**Result: gate FAIL / not promoted** (GermanQuAD guardrail regresses −0.0285 with 16.9% catastrophic
per-query drops). See `V5_RESULTS.md`.
