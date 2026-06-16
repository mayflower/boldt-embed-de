# Frozen-policy promotion gate

Decides whether the **frozen** bounded rerank policy `configs/policies/bounded_margin_override_v1.json`
may be promoted from Experimental. **The decision is about the policy, never the raw model.** Pure
stdlib, no ML — it reads the per-set eval reports produced by `eval_policy_gate_v5.py`.

- Eval: `scripts/eval_policy_gate_v5.py`
- Gate: `scripts/check_policy_promotion_gate.py` (`promotion_gate(reports, tuned_on_guardrail=…)`)
- Outputs: `outputs/v5-small-rag/policy/{eval_<set>.json, promotion_gate.json, promotion_gate.md}`

## Eval sets and roles

| set | role | used by gate |
|---|---|---|
| `webfaq` | primary | yes — lift target |
| `near_ceiling` | primary (near-ceiling guardrail) | yes — must do no harm + beat raw |
| `germanquad` | guardrail | yes — must do no harm + beat raw |
| `dt_test` | guardrail | yes — must do no harm |
| `local_rag` | primary (optional) | reported |
| `gerdalir` | diagnostic | **ignored by the gate** |

Each set is a directory `<set>.jsonl` (or `<set>_scored.jsonl`) of scored candidate lists. For every
list the eval computes first-stage / raw-rerank / policy nDCG@10, `policy_delta`, `raw_delta`,
`catastrophic_drop_rate` (per-query policy−first-stage ≤ −0.2), action rates
(abstain/lock/blend/margin_override), the max-downshift distribution, medium+hard lift, and the
no_room delta.

## Gate conditions (all must hold)

1. WebFAQ `policy_delta ≥ +0.05`
2. near_ceiling `catastrophic ≤ 0.03` and `policy_delta ≥ −0.005`
3. GermanQuAD `catastrophic ≤ 0.03` and `policy_delta ≥ −0.005`
4. DT-test `catastrophic ≤ 0.02` and `policy_delta ≥ −0.005`
5. policy **beats** raw-rerank on GermanQuAD **and** near_ceiling (`policy_delta > raw_delta`)
6. raw always-rerank is **never** recommended, and a `raw_rerank`-mode report can never pass
7. a policy tuned on a guardrail fails closed — `TUNED_ON_GUARDRAIL` marker in the eval dir, or any
   report with `tuned_on ∈ {germanquad, dt_test}`. GerDaLIR is diagnostic and ignored entirely.

## Why these sets

GermanQuAD and DT-test stay **guardrails** — they are never tuned on. The held-out near-ceiling set
(716 train-disjoint WebFAQ lists, see `near-ceiling-guardrails.md`) is where the policy must prove it
does no harm on already-perfect first-stage rankings — the remaining risk the GermanQuAD numbers
alone can't isolate.

## Run

```bash
python scripts/eval_policy_gate_v5.py \
  --policy configs/policies/bounded_margin_override_v1.json \
  --eval-dir outputs/v5-small-rag/policy/eval_inputs \
  --output-dir outputs/v5-small-rag/policy
python scripts/check_policy_promotion_gate.py \
  --eval-dir outputs/v5-small-rag/policy \
  --output outputs/v5-small-rag/policy/promotion_gate.json \
  --markdown outputs/v5-small-rag/policy/promotion_gate.md
```

`eval_inputs/` holds the conservative-scored eval lists named per set (`webfaq.jsonl`,
`near_ceiling.jsonl` = the guardrail artifact, `germanquad.jsonl`, `dt_test.jsonl`). A `pass` means
**promotable with the bounded policy only**; raw always-rerank remains not recommended regardless.
