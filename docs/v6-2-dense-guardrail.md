# v6.2 — dense guardrail recovery (keep Recall@50, restore GermanQuAD)

v6.1 hit its WebFAQ objective (Recall@50 0.883 → 0.933) but its dense gate **failed** by 0.002 on the
GermanQuAD guardrail (nDCG@10 0.886 → **0.878** < 0.88). v6.2 aims to **recover the GermanQuAD nDCG
while keeping the Recall@50 win** — **DENSE-ONLY, no reranker.**

## Diagnosis

v6.1's rank-promotion triplets are **100% WebFAQ** (the dense top-50 blockers were mined on WebFAQ),
so the gradient over-rotated toward FAQ-style retrieval and drifted away from the GermanQuAD-style
QA-passage distribution — costing −0.008 GermanQuAD nDCG.

## Fix (no leakage)

**GermanQuAD is a public eval benchmark and is NEVER trained on.** v6.2 instead **upweights
GermanQuAD-*style* NON-eval pairs** already in the training set — `wiki_non_eval` (DT-de-dpr train,
German Wikipedia QA) + `qa_passage_non_eval` — so the contrastive signal preserves the QA-passage
distribution during the WebFAQ rank-promotion. The upweighted pair set
(`data/processed/v6_2/rag_pairs_guardrail_upweighted.jsonl`) raises the wiki/QA share **0.47 → 0.64**
(verified 0 public-eval leakage via `leakage_reason`). The WebFAQ rank-promotion triplets are
unchanged (`data/processed/v6_1/dense_top50_hardnegatives.jsonl`).

Train **from the v6 checkpoint** (GermanQuAD 0.886, above the floor) with CMNRL → Matryoshka[1024…64]
+ rank-promotion (the v6.1 trainer, `scripts/train_v6_1_dense_top50.py`), so v6.2 is a clean re-do of
v6.1 with a guardrail-preserving mix rather than continuing from v6.1's already-drifted state.

## Run

```bash
python scripts/train_v6_1_dense_top50.py \
  --base-model outputs/v6-dense-rag/checkpoints/boldt-dense-rag-v6 \
  --train-pairs data/processed/v6_2/rag_pairs_guardrail_upweighted.jsonl \
  --hard-negatives data/processed/v6_1/dense_top50_hardnegatives.jsonl \
  --output outputs/v6-2-dense-guardrail/checkpoints/boldt-dense-rag-v6-2 \
  --max-steps 1000 --batch-size 64 --bf16 --run-id v6-2-dense-guardrail
python scripts/eval_v6_1_dense_top50.py --models dense-v6.2 \
  --summary outputs/v6-2-dense-guardrail/dense_eval_summary.json
python scripts/check_v6_1_dense_gate.py --model dense-v6.2 \
  --summary outputs/v6-2-dense-guardrail/dense_eval_summary.json \
  --output outputs/v6-2-dense-guardrail/dense_gate.json --markdown outputs/v6-2-dense-guardrail/dense_gate.md
```

## Promotion

Same dense gate as v6.1 (`scripts/check_v6_1_dense_gate.py`): promote only if **WebFAQ Recall@50 ≥
0.90 AND Recall@100 ≥ 0.96 AND GermanQuAD nDCG@10 ≥ 0.88 AND DT-test ≥ 0.94 AND 256-retention ≥ 0.95**.
This is an **honest re-attempt** — whether the rebalance recovers GermanQuAD without dropping
Recall@50 below 0.90 is uncertain until measured; the gate decides. **No reranker work** until the
dense gate passes.

## Results

Filled from `outputs/v6-2-dense-guardrail/dense_gate.{json,md}` after the run.
