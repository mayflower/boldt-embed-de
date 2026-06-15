# Conservative reranker training (rank-preservation loss)

Trains a v5 reranker that **penalizes unnecessary reordering on near-ceiling lists**, to cut the
churn that made v5 always-rerank degrade GermanQuAD. No new teacher calls — uses the existing v5
teacher-scored candidate lists.

- Loss: `src/boldt_embed/rank_preservation_loss.py` + `reranker_modern.train_conservative_listwise_reranker`
- CLI: `scripts/train_v5_rag_reranker_conservative.py`

## Objective

```
total = listwise_teacher_kl + pairwise_margin + pointwise_confident_bce
        + lambda_preserve * rank_preservation_loss
```

Listwise KL is primary; the **rank-preservation** term is applied ONLY to high-first-stage-
confidence lists.

### Rank-preservation loss
For every first-stage-ordered pair (i above j) in a high-confidence list, if the teacher does
**not** advantage j over i by ≥ `justify_margin` (default 2.0), penalize the student for scoring j
above i: `relu(student_j − student_i)`. So:
- **0** when the student preserves the first-stage order;
- **large** when it moves a doc (incl. the first-stage top1) above a better-first-stage doc without
  teacher support;
- **allowed** (no penalty) when the teacher margin justifies the move.

It uses only teacher scores + first-stage ranks (training signals) — **never eval qrels**.

### High-confidence detection (observable features only)
`is_high_confidence` uses `first_stage_top1_top2_gap`, `first_stage_score_entropy`, and
`candidate_source_agreement` — the same observable features as the abstention policy, **no qrels**.
Lists with first-stage gap ≥ the chosen percentile (default 60th of the training set) are treated
as high-confidence.

## CLI

```bash
python scripts/train_v5_rag_reranker_conservative.py \
  --candidate-lists outputs/v5-small-rag/teacher/rag_train_scored.jsonl \
  --output outputs/v5-small-rag/checkpoints/boldt-rag-reranker-v5-conservative \
  --lambda-preserve 0.2 --bf16 --gradient-checkpointing --run-id v5-reranker-conservative
```

`--dry-run` writes the loss plan + run card with no torch.

## Result (EXECUTED 2026-06-15, RTX A6000)

Trained on 5,658 lists (2,265 = 40% high-confidence; FAQ share 0.217); mean preservation loss
0.220. Scored the SAME fixed eval lists and ran the SAME hardness gate. GermanQuAD catastrophic-drop
rate across approaches (apples-to-apples):

| approach | GermanQuAD overall Δ | GermanQuAD catastrophic | WebFAQ overall Δ |
|---|--:|--:|--:|
| v5 raw (always-rerank) | −0.0285 | 0.169 | +0.1665 |
| abstain only | −0.0016 | 0.103 | +0.1285 |
| **conservative only** | **+0.0094** | 0.122 | +0.1379 |
| **conservative + abstain** | **+0.0243** | **0.074** | +0.0975 |

**The conservative objective reduces catastrophic drops** (0.169 → 0.122 alone, 0.074 combined with
abstention) and turns GermanQuAD overall **positive** (+0.0094 / +0.0243), while keeping WebFAQ and
DT-test healthy. **It is still evaluated by the same gate, and the gate still FAILS** the strict bars
(germanquad catastrophic 0.074 > 0.03; dt_test marginally below always-rerank). So the reranker
remains **Experimental — not recommended**, but the failure is now a *small residual near-ceiling
churn*, not the original −0.07 regression.

## Acceptance

- ✅ Conservative reranker **reduces catastrophic drops** (0.169 → 0.122 / 0.074).
- ✅ Still evaluated by the **same hardness-aware gate + abstention evaluation**.
- Not promoted: the residual GermanQuAD catastrophic rate (0.074) still exceeds the 0.03 bar; next
  levers are a higher `lambda_preserve`, a tighter high-confidence percentile, and set-aware
  abstention that doesn't give up DT-test's existing lift.

## Preservation grid (lp04/lp06/lp08) — negative training result (2026-06-15)

We then trained the higher-λ / stricter-percentile variants (`--preserve-top-k 3`,
`--teacher-margin-override 3.0`) to test whether stronger preservation fixes catastrophic at the
MODEL level (so raw always-rerank is safe, removing the policy-selection dependency).

| checkpoint | λ / hc-pct | RAW always-rerank GQ catastrophic | RAW GQ Δ | RAW WebFAQ Δ | bounded(margin_override) GQ catastrophic | bounded gate |
|---|---|--:|--:|--:|--:|:--|
| conservative (orig) | 0.2 / 0.60 | 0.123 | +0.009 | +0.140 | 0.015 | pass |
| lp04 | 0.4 / 0.70 | 0.175 | −0.029 | +0.156 | 0.028 | pass |
| lp06 | 0.6 / 0.75 | 0.137 | −0.002 | +0.144 | 0.019 | pass |
| lp08 | 0.8 / 0.80 | 0.112 | +0.018 | +0.196 | 0.015 | pass |

**Conclusion: more preservation does not make raw always-rerank safe.** No λ brings raw GermanQuAD
catastrophic near 0.03 (lp04 is *worse*); preservation protects WebFAQ-gap-defined high-confidence
lists that don't transfer to GermanQuAD's near-ceiling lists. WebFAQ lift did not collapse (no
variant is "too conservative"). Bounded `margin_override` already passes on the original checkpoint,
so **retraining did not change the deployable answer** — the original conservative checkpoint +
bounded policy remains the candidate; lp04/lp06/lp08 are not promoted. The reranker stays
**Experimental** until the bounded policy is frozen and validated on a held-out near-ceiling
guardrail. Full numbers: `outputs/v5-small-rag/grid/grid_comparison.{json,md}`.
