# v3 domain-quality gates

**Make domain quality visible before training.** v2 was nominally 7-domain, but the teacher
rejected the synthetic admin/FAQ/legal queries (admin 4.8%, faq 5.7% accepted), so the effective
training set was ~web+wiki — and nothing flagged it. v3 runs this analysis between teacher
scoring and training and **blocks training** when the real domains aren't actually present and
teacher-accepted.

Module: `boldt_embed.domain_quality` · CLI: `scripts/analyze_domain_quality.py` · outputs:
`outputs/v3-real-domain/domain_quality.{json,md}`.

## Inputs

- candidate JSONL (the raw universe, with `domain`/`source`/`license` provenance),
- teacher cache JSONL or `*.manifest.json` (per-row `reranker_score`/`embedding_score`/`positive`),
- v3 config (optional `domain_quality_gates` block; else built-in defaults),
- source manifest (optional — used to flag supplemental + disallowed sources).

## Metrics computed

1. raw candidates by domain / source / license
2. teacher-validated positives (accepted) by domain / source / license
3. acceptance rate per domain = accepted / raw
4. effective training distribution (share of accepted positives per domain)
5. synthetic vs real ratio per domain (on accepted positives)
6. median embedding score per domain
7. median reranker score per domain
8. suspicious positives per domain (positives the teacher scored below threshold)
9. unknown-license rows
10. disallowed-source rows (source `allowed_for_training=false`)

A row counts as **accepted** when `positive != false` and `reranker_score >= --reranker-threshold`
(default 2.0, the v2 validation threshold). A row is **synthetic** when flagged
(`synthetic`/`generated`), inherited-license (`license_origin=inherited`), or from a `supplemental`
source.

## Gates (failure → non-zero exit → training blocked)

| gate | rule | default |
|---|---|---|
| `license_unknown_rows_zero` | unknown-license rows == 0 | 0 |
| `disallowed_source_rows_zero` | disallowed-source rows == 0 | 0 |
| `real_domain_min_raw` / `real_domain_min_accepted` | each real domain ≥ floor (raw **and** accepted) | 5000 each |
| `real_domain_synthetic_share` | synthetic share within each real domain ≤ cap | 0.25 |
| `real_domain_acceptance_rate` | teacher acceptance per real domain ≥ floor | 0.35 |
| `effective_web_wiki_share` | accepted web+wiki share ≤ cap | 0.65 |

Real domains: `faq_real`, `admin_real`, `legal_adjacency_real_no_eval_overlap`.

**Non-blocking review flags:** a *non-real* domain below the acceptance floor is flagged for
review (not failed). **Legal-transfer claim:** if `legal_adjacency_real_no_eval_overlap` accepted
is below its floor, `can_claim_legal_transfer_from_data` is **false** — do not attribute any
GerDaLIR improvement to domain data.

## Default gate config

```json
{
  "min_real_domain_accepted": {
    "faq_real": 5000, "admin_real": 5000, "legal_adjacency_real_no_eval_overlap": 5000
  },
  "max_synthetic_share_for_real_domains": 0.25,
  "max_effective_web_wiki_share": 0.65,
  "min_teacher_acceptance_rate": 0.35
}
```

Override by adding a `domain_quality_gates` block to the v3 config
(`configs/experiments/v3_real_domain_generalization.json`).

## CLI

```bash
python scripts/analyze_domain_quality.py \
  --candidates data/processed/candidates_v3.jsonl \
  --teacher-cache outputs/v3-real-domain/teacher-cache/qwen3_v3.manifest.json \
  --config configs/experiments/v3_real_domain_generalization.json \
  --source-manifest configs/data_sources_v3.json \
  --output outputs/v3-real-domain/domain_quality.json \
  --markdown outputs/v3-real-domain/domain_quality.md
# exit 0 = gates pass; exit 1 = a gate failed -> do not train.
```

## Why this exists

This is the gate that would have stopped the v2 run from quietly training a "multi-domain" model
that was really web+wiki. With v3's real corpora it stays green only when real admin/FAQ/legal
data is genuinely present **and** survives teacher validation — otherwise it fails loudly and the
training step refuses to run. See `docs/v3-real-domain-generalization-plan.md` and
`docs/data/v3-real-domain-sources.md`.
