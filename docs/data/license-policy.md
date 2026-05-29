# License Policy

Implements ADR-004. Three license layers are tracked **separately**:

1. **Code** — Apache-2.0 (`LICENSE`).
2. **Base weights** — `Boldt/Boldt-DC-350M` = `apache-2.0` (verified 2026-05-29).
3. **Training data** — per-source; only the allowlist below.

## Allowed training-data licenses (`boldt_embed.data.ALLOWED_LICENSES`)
`cc0-1.0`, `cc-by-4.0`, `cc-by-sa-4.0`, `apache-2.0`, `mit`, `public-domain`, `synthetic`.

- **Attribution (CC-BY):** keep source attribution in the dataset card.
- **Share-alike (CC-BY-SA):** a published dataset derived from SA sources inherits SA terms;
  the dataset card states this.
- **Synthetic:** our own generations; record generator model + prompt version (reproducibility).

## Derivative weights
Trained weights are intended Apache-2.0, **contingent** on every training dataset being
license-compatible with redistribution. Do not state a weights license on a model card until
this is confirmed for all sources (release checklist gate).

## Enforcement
`scripts/validate_data_schema.py` fails the build on any `license` outside the allowlist
(`data.check_licenses`). Per-source license + URL + retrieval date live in `data-sources.md`.
