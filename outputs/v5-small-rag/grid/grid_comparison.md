# v5 conservative grid (stronger preservation) — results (2026-06-15, RTX A6000)

Three variants trained (preserve_top_k=3, teacher_margin_override=3.0), scored on the same fixed
eval lists, compared apples-to-apples to the original conservative checkpoint.

| checkpoint | λ / hc-pct | RAW always-rerank GQ catastrophic | RAW GQ Δ | RAW WebFAQ Δ | bounded(margin_override) GQ catastrophic | bounded gate |
|---|---|--:|--:|--:|--:|:--|
| conservative (orig) | 0.2 / 0.60 | 0.123 | +0.009 | +0.140 | 0.015 | pass |
| lp04 | 0.4 / 0.70 | 0.175 | −0.029 | +0.156 | 0.028 | pass |
| lp06 | 0.6 / 0.75 | 0.137 | −0.002 | +0.144 | 0.019 | pass |
| lp08 | 0.8 / 0.80 | 0.112 | +0.018 | +0.196 | 0.015 | pass |

**Verdict: the stronger-preservation grid did NOT solve catastrophic at the model level.** Raw
always-rerank GermanQuAD catastrophic stays 0.11–0.18 for all λ (lp04 worse); no variant approaches
the 0.03 bar without a policy. Root cause: preservation targets WebFAQ-gap-defined high-confidence
lists, which do not transfer to GermanQuAD's near-ceiling lists — the same transfer gap seen with
fitted policy thresholds. WebFAQ lift did NOT collapse (lp08 even improved it), so none are "too
conservative". The deployable fix remains the **bounded margin_override policy**, which already
passed the gate on the ORIGINAL checkpoint. Retraining was not the answer (consistent with the
catastrophic-drop analysis: 100% policy-fixable). **Recommendation unchanged — reranker stays
Experimental; no checkpoint promoted.**
