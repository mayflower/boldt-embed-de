# Leakage & Safety Policy

Implements ADR-004 (leakage) and ADR-009 (train/eval split), plus the PII/safety filtering
dimension from prompt 04.

## Benchmark leakage
- **Eval registry:** GermanDPR / GermanQuAD (test), MMTEB German tasks, and any private eval set.
- **Check:** `boldt_embed.data.find_leakage(train, eval_texts)` flags exact (normalized) and
  near-duplicate (token-Jaccard ≥ 0.9) overlap between training `query`/`positive` and eval text.
- **Rule:** public **test** splits never enter training or the dev split. Dev is carved from
  training-domain data only (ADR-009).

## Deduplication
- Within-corpus and cross-corpus dedup by normalized text (umlaut/ß folded) before training.
- Near-duplicate positives across sources are collapsed to avoid trivial in-batch negatives.

## PII (safety)
- `boldt_embed.data.scan_pii` detects emails, German IBANs, phone numbers, and IPv4 addresses
  across `query`/`positive`/`negatives`.
- **Rule:** records with PII are dropped or redacted before training; the gate fails on any hit
  in shipped data. Patterns are conservative to avoid flagging legal refs (`§ 543`) or years.

## Filtering dimensions (prompt 04)
language quality · license compatibility · **PII/safety** · deduplication · positive relevance ·
hard-negative plausibility · **benchmark leakage** — each enforced by `validate_data_schema.py`
or documented as a manual review step here.
