---

# Claude Code prompt set: EmbedFilter / spectral Matryoshka for `mayflower/boldt-embed-de`

## Prompt 0 — Inspect and produce the implementation plan

```markdown
You are working in `mayflower/boldt-embed-de`.

Goal: add a gated v7 experiment inspired by arXiv 2606.07502, “Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings,” without overclaiming results.

Core idea to implement:
- Build an EmbedFilter basis from the model unembedding matrix.
- Use `torch.linalg.svd(W, full_matrices=False)` on `W = lm_head.weight` / output embeddings, shape `[vocab, hidden]`.
- `Vh` rows are right singular directions ordered by descending singular values.
- For hidden size `H` and filter ratio `tau`, keep `K = H // tau` central directions.
- Let `left = (H - K) // 2`, `right = left + K`.
- Store `basis = Vh[left:right].T`, shape `[H, K]`.
- Apply to pooled embeddings as `z = pooled @ basis`, then L2-normalize.
- Compare this against the current prefix Matryoshka truncation.

Repo constraints:
- Importable core and validation gates must stay Python stdlib only.
- `torch`, `transformers`, and `sentence_transformers` must be lazy imports only.
- Do not commit model weights, large datasets, `.pt` matrices under outputs, or secrets.
- Never claim a benchmark result unless the command was run and saved under `outputs/` with metadata.
- German-first and leakage/licensing policies must remain visible.

First task:
1. Inspect `CLAUDE.md`, `README.md`, `src/boldt_embed/matryoshka.py`, `src/boldt_embed/model_causal.py`, `src/boldt_embed/model_bidirectional.py`, `scripts/dense_retrieve.py`, `src/boldt_embed/train_modern.py`, and the current validation scripts.
2. Produce a short implementation plan with files to touch, tests to add, commands to run, and risks.
3. Do not edit files yet.
```

## Prompt 1 — Add stdlib-safe EmbedFilter core

````markdown
Implement the stdlib-safe core for EmbedFilter.

Add `src/boldt_embed/embed_filter.py` with:
- `EmbedFilterSpec` dataclass:
  - `hidden_dim: int`
  - `tau: int`
  - `keep_dim: int`
  - `left: int`
  - `right: int`
  - `strategy: str = "bulk_center"`
- `select_bulk_slice(hidden_dim: int, tau: int) -> EmbedFilterSpec`
  - Require `hidden_dim > 0`.
  - Require `tau in {1, 2, 4, 8, 16}` for now.
  - Require `hidden_dim % tau == 0`.
  - `tau=1` should keep all dims: left=0, right=hidden_dim.
  - For tau>1, keep the centered contiguous slice.
- `validate_embed_filter_metadata(meta: dict) -> list[str]`
  - stdlib only.
  - Check model name, hidden dim, tau, keep dim, left/right bounds, source matrix, and artifact format.
- `metadata_for_spec(...) -> dict`
  - Return JSON-serializable metadata for run cards and artifact sidecars.
- No torch import at module import time.

Add unit tests under `tests/test_embed_filter.py`:
- `select_bulk_slice(1024, 1/2/4/8/16)` returns keep dims 1024/512/256/128/64 and centered slices.
- Invalid tau and non-divisible dimensions raise `ValueError`.
- `validate_embed_filter_metadata` catches wrong dims and bad bounds.
- Importing `boldt_embed.embed_filter` must not import `torch`.

Update `src/boldt_embed/__init__.py` docstring only if needed; do not add heavy imports.

Run:
```bash
python -m unittest discover -s tests
python scripts/validate_repo.py --format markdown
````

Report:
Files changed · Commands run · Validation · Benchmark: not run · Risks.

````

## Prompt 2 — Add the GPU artifact builder

```markdown
Add a lazy-ML script to build EmbedFilter projection artifacts.

Create `scripts/build_embed_filter.py`.

CLI:
```bash
python scripts/build_embed_filter.py \
  --model Boldt/Boldt-DC-350M \
  --tau 2 \
  --out outputs/embedfilter/boldt-dc-350m_tau2
````

Requirements:

* Lazy import `torch` and `transformers` inside `main()` only.
* Load model with `AutoModelForCausalLM.from_pretrained`.
* Prefer `model.get_output_embeddings().weight`; fallback to `model.lm_head.weight`; final fallback to tied input embeddings only with a clear metadata warning.
* Validate hidden dim against `select_bulk_slice`.
* Compute SVD on CPU or GPU according to `--device auto|cpu|cuda`, default `auto`.
* Always cast matrix to float32 for SVD unless `--svd-dtype` overrides.
* Save:

  * `<out>/basis.pt` containing a tensor `[hidden_dim, keep_dim]`.
  * `<out>/metadata.json` containing model, hidden_dim, vocab_size, tau, keep_dim, left, right, singular value stats, source matrix, torch/transformers versions, command, git commit if available, timestamp, and warning list.
* Add `.gitignore` rules if needed so generated `.pt` artifacts under `outputs/embedfilter/` are not committed.
* Add `--dry-run`:

  * Loads no model.
  * Prints planned spec from `--hidden-dim` and `--tau`.
  * Imports no torch/transformers.
* Add a lightweight JSON schema or validation helper only if it remains stdlib-safe.

Tests:

* Unit-test dry-run planner without torch.
* Unit-test metadata validation.
* Do not unit-test real SVD in CI.

Run:

```bash
python scripts/build_embed_filter.py --dry-run --hidden-dim 1024 --tau 2 --out outputs/embedfilter/dry
python -m unittest discover -s tests
python scripts/validate_repo.py --format markdown
```

Report:
Files changed · Commands run · Validation · Benchmark: not run · Risks.

````

## Prompt 3 — Apply EmbedFilter in model wrappers and dense retrieval

```markdown
Wire an optional EmbedFilter projection into existing embedding paths.

Targets:
- `src/boldt_embed/model_causal.py`
- `src/boldt_embed/model_bidirectional.py`
- `scripts/dense_retrieve.py`
- Add any helper functions to `src/boldt_embed/embed_filter.py` only if they preserve stdlib import safety.

Implementation details:
1. Add lazy torch helpers:
   - `load_embed_filter_basis(path_or_dir, *, expected_hidden_dim=None)`.
   - Accept either a directory containing `basis.pt` + `metadata.json` or a direct `.pt` path.
   - Validate metadata if present.
   - Return `(basis_tensor, metadata_dict)`.
2. Apply projection before final normalization where possible:
   - pooled hidden `[B, H]`
   - if filter present: `pooled = pooled @ basis.to(pooled.device, pooled.dtype)`
   - then L2-normalize.
3. In `CausalEmbedder.encode(...)`:
   - Add optional keyword `embed_filter: Optional[str] = None`.
   - If `dim` and `embed_filter` are both provided, raise `ValueError`; they are competing reduction methods.
4. In `BidirectionalEmbedder.encode(...)`:
   - Same optional keyword and conflict behavior.
5. In `scripts/dense_retrieve.py`:
   - Add `--embed-filter PATH`.
   - If set, call `SentenceTransformer.encode(..., normalize_embeddings=False)`, apply the basis to corpus and query tensors, then normalize.
   - If not set, preserve existing behavior exactly.
   - Ensure corpus/query projected dims match.
   - Include embed-filter metadata in the output sidecar if the script writes one; otherwise add a small JSON report next to `--out`.

Safety:
- Do not change default behavior.
- Do not import torch in stdlib validation paths.
- Do not silently use both prefix dim truncation and EmbedFilter.

Tests:
- Unit-test CLI argument parsing or helper planning where feasible without torch.
- Add a tiny torch-only test only if it is skipped automatically when torch is unavailable.
- Test that setting both `dim` and `embed_filter` raises.

Run:
```bash
python -m unittest discover -s tests
python scripts/validate_repo.py --format markdown
python scripts/run_smoke_tests.py --format markdown
````

Report:
Files changed · Commands run · Validation · Benchmark: not run · Risks.

````

## Prompt 4 — Add an EmbedFilter sweep/evaluation harness

```markdown
Add an evaluation harness to compare:
- current full 1024-d embeddings,
- current prefix Matryoshka dims,
- EmbedFilter tau values producing equivalent dims.

Create:
- `configs/experiments/v7_embedfilter.json`
- `scripts/eval_embed_filter_sweep.py`
- `docs/v7-embedfilter-plan.md`

Config should include:
```json
{
  "experiment": "v7_embedfilter",
  "base_model": "Boldt/Boldt-DC-350M",
  "candidate_embedder": "outputs/checkpoints/<fill-me>",
  "taus": [1, 2, 4, 8, 16],
  "compare_prefix_dims": [1024, 512, 256, 128, 64],
  "active_eval_sets": ["webfaq_heldout", "local_rag", "germanquad", "dt_test"],
  "diagnostic_eval_sets": ["gerdalir"],
  "primary_metrics": ["nDCG@10", "Recall@100"],
  "promotion_policy": "advisory_only_until_real_outputs_exist"
}
````

Evaluation script requirements:

* `--dry-run` must parse config, verify artifact paths or planned outputs, and import no ML packages.
* Real mode may import torch/sentence_transformers lazily.
* Accept prebuilt artifact directories from `scripts/build_embed_filter.py`.
* Produce:

  * `outputs/v7-embedfilter/sweep.json`
  * `outputs/v7-embedfilter/sweep.md`
  * a run card with command, commit, model, dataset/eval set names, artifact paths, hardware, timestamps.
* Report each row:

  * method: `full`, `prefix`, `embedfilter`
  * dim
  * tau if applicable
  * eval set
  * nDCG@10, Recall@100, MRR@10 if available
  * bytes/vector estimate
  * relative delta vs full and vs prefix same dim
* Do not mark promoted automatically unless real metrics exist and gates pass.

Advisory gate:

* tau=2 / 512-d passes only if mean nDCG@10 and Recall@100 are within 0.005 absolute of full 1024 on active eval sets, or improve.
* tau=4 / 256-d passes only if it beats or matches prefix-256 on mean nDCG@10 and Recall@100.
* GermanQuAD and DT-test near-ceiling guardrails: no active eval regression worse than -0.005 unless explicitly marked diagnostic.
* WebFAQ/local RAG are primary for product RAG recall.
* GerDaLIR is diagnostic-only unless the docs say product scope changed.

Docs:

* Explain that this is an unembedding-spectrum postprocessor, not a new trained model.
* Explain that outputs cannot be claimed until saved under `outputs/`.
* Explain why this competes against Matryoshka prefix truncation.

Run:

```bash
python scripts/eval_embed_filter_sweep.py --config configs/experiments/v7_embedfilter.json --dry-run
python -m unittest discover -s tests
python scripts/validate_repo.py --format markdown
```

Report:
Files changed · Commands run · Validation · Benchmark: dry-run only · Risks.

````

## Prompt 5 — Add unembedding-lens diagnostics for German stop-token collapse

```markdown
Add a diagnostic script to see whether Boldt embeddings exhibit the paper’s “frequent-token lens” pattern.

Create `scripts/diagnose_unembedding_lens.py`.

Purpose:
- Encode a small German diagnostic text set.
- Project pooled embeddings through the unembedding matrix.
- Show top decoded tokens before and after applying an EmbedFilter basis.
- Summarize how many top-k tokens are punctuation/stopwords/subword fragments versus content-bearing German terms.

Requirements:
- Lazy import torch/transformers only in real mode.
- `--dry-run` imports no ML packages and prints planned diagnostics.
- Inputs:
  - `--model`
  - `--texts data/samples/embedfilter_diagnostics_de.jsonl`
  - `--embed-filter outputs/embedfilter/...`
  - `--top-k 20`
  - `--out outputs/v7-embedfilter/unembedding_lens.json`
- Add a small permissive sample file with German diagnostic texts, no benchmark content.
- Include a small German stopword/punctuation list in the script or a stdlib data file.
- Output JSON and Markdown:
  - top tokens before/after per text,
  - stopword/punctuation ratio before/after,
  - anisotropy proxy: mean pairwise cosine on sample embeddings before/after.
- This is diagnostic only; it must not be used as a quality claim.

Run:
```bash
python scripts/diagnose_unembedding_lens.py --dry-run --model Boldt/Boldt-DC-350M --top-k 20
python -m unittest discover -s tests
python scripts/validate_repo.py --format markdown
````

Report:
Files changed · Commands run · Validation · Benchmark: not run · Risks.

````

## Prompt 6 — Optional train-time regularizer, behind a hard-off flag

```markdown
Only do this after the post-processing path is implemented and validated.

Add an optional train-time edge-spectrum suppression regularizer, disabled by default.

Motivation:
The paper is post-processing-first, but the insight suggests a possible training regularizer: discourage pooled embeddings from carrying energy in the dropped edge singular subspace.

Implementation constraints:
- This must be opt-in only.
- Default configs must remain unchanged.
- The dry-run loss plan must show when it is enabled.
- No claims unless a real run is executed and saved.

Design:
- Extend `configs/student_training_2026.json` only through a new optional experiment config, not the default production config:
  - `configs/experiments/v7_embedfilter_training.json`
- Add fields:
```json
{
  "edge_spectrum_regularizer": {
    "enabled": false,
    "embed_filter_artifact": "outputs/embedfilter/boldt-dc-350m_tau4",
    "lambda": 0.0,
    "apply_to": "pooled_embeddings",
    "normalize_before": false
  }
}
````

* In `src/boldt_embed/train_modern.py`, add planning support in stdlib-only functions:

  * dry-run reports the regularizer as disabled/enabled.
* In ML path only:

  * Load the kept bulk basis and derive an edge basis or load/store both from artifact builder.
  * Regularizer = mean squared norm of projection onto edge basis.
  * Add to loss only if enabled and lambda > 0.
* Ensure the implementation works with LoRA and gradient checkpointing.
* Add tests for dry-run planning only.

Do not run real training unless explicitly asked. Add a note to docs that this is experimental and not from the paper’s evaluated method.

Run:

```bash
python scripts/train_modern_embedder.py --dry-run --student-config configs/student_training_2026.json
python -m unittest discover -s tests
python scripts/validate_repo.py --format markdown
```

Report:
Files changed · Commands run · Validation · Benchmark: not run · Risks.

````

## Prompt 7 — Documentation, release gates, and model-card honesty

```markdown
Update documentation and validation so EmbedFilter cannot be overclaimed.

Docs to update/add:
- `docs/v7-embedfilter-plan.md`
- `docs/benchmark-report.md` only with “planned” or “dry-run” status unless real outputs exist.
- `AUDIT.md` only if there is a real executed result or a new known gap.
- Model cards only if real metrics exist; otherwise do not mention EmbedFilter as recommended.

Validation:
- Update `scripts/validate_repo.py` required file lists if new scripts/docs/configs must be present.
- Add a lightweight checker `scripts/check_embedfilter_gate.py`:
  - Reads `outputs/v7-embedfilter/sweep.json`.
  - Fails if missing real metrics when `--require-real` is set.
  - Emits advisory pass/fail otherwise.
  - Never fabricates metrics.
- Integrate into `scripts/validate_release_2026.py` only as an advisory v7 gate unless explicitly asked to make it blocking.

README:
- Add a short “v7 EmbedFilter experiment” section only after implementation exists.
- Phrase it as:
  - “implemented as an experimental postprocessor”
  - “no production recommendation until the v7 gate passes”
  - “competes against prefix Matryoshka at the same dimensions”
- Do not say it improves quality unless `outputs/v7-embedfilter/sweep.json` shows that.

Run:
```bash
python scripts/check_embedfilter_gate.py --help
python -m unittest discover -s tests
python scripts/validate_repo.py --format markdown
python scripts/run_smoke_tests.py --format markdown
````

Report:
Files changed · Commands run · Validation · Benchmark: not run unless real sweep executed · Risks.

````

## Prompt 8 — Real run recipe, only when GPU/model access is available

```markdown
Run the real v7 EmbedFilter experiment. Do not edit code unless a bug is found.

Preflight:
```bash
git status --short
python scripts/validate_repo.py --format markdown
python -m unittest discover -s tests
nvidia-smi
````

Build artifacts:

```bash
python scripts/build_embed_filter.py \
  --model Boldt/Boldt-DC-350M \
  --tau 1 \
  --out outputs/embedfilter/boldt-dc-350m_tau1

python scripts/build_embed_filter.py \
  --model Boldt/Boldt-DC-350M \
  --tau 2 \
  --out outputs/embedfilter/boldt-dc-350m_tau2

python scripts/build_embed_filter.py \
  --model Boldt/Boldt-DC-350M \
  --tau 4 \
  --out outputs/embedfilter/boldt-dc-350m_tau4

python scripts/build_embed_filter.py \
  --model Boldt/Boldt-DC-350M \
  --tau 8 \
  --out outputs/embedfilter/boldt-dc-350m_tau8

python scripts/build_embed_filter.py \
  --model Boldt/Boldt-DC-350M \
  --tau 16 \
  --out outputs/embedfilter/boldt-dc-350m_tau16
```

Run diagnostics:

```bash
python scripts/diagnose_unembedding_lens.py \
  --model Boldt/Boldt-DC-350M \
  --texts data/samples/embedfilter_diagnostics_de.jsonl \
  --embed-filter outputs/embedfilter/boldt-dc-350m_tau4 \
  --top-k 20 \
  --out outputs/v7-embedfilter/unembedding_lens.json
```

Run sweep:

```bash
python scripts/eval_embed_filter_sweep.py \
  --config configs/experiments/v7_embedfilter.json \
  --out outputs/v7-embedfilter/sweep.json
```

Gate:

```bash
python scripts/check_embedfilter_gate.py \
  --sweep outputs/v7-embedfilter/sweep.json \
  --require-real
```

After run:

1. Summarize full vs prefix vs EmbedFilter at 512/256/128/64 dims.
2. State whether each gate passed.
3. Update docs only with exact saved results.
4. Do not call it production-ready unless the gate passes.
5. Include hardware, commit, commands, and output paths.

Report:
Files changed · Commands run · Validation · Benchmark outputs · Gate verdict · Risks · Working tree.

```

---

The highest-value part is **Prompt 4**: it turns the paper into a fair head-to-head against the repo’s existing Matryoshka approach. The current repo already measured that 256-d prefix Matryoshka keeps about 97% of full GermanQuAD quality, so EmbedFilter only matters if it beats or matches that at the same dimension while helping active RAG recall. 
```

[1]: https://arxiv.org/pdf/2606.07502 "Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings"

