# v7 — EmbedFilter / spectral Matryoshka (experiment plan)

**Status: experimental postprocessor. No production recommendation until the v7 gate passes on
real saved outputs.** Inspired by arXiv 2606.07502 ("Your UnEmbedding Matrix is Secretly a Feature
Lens for Text Embeddings").

## Idea

Build a projection basis from the model's **unembedding matrix** `W = lm_head.weight` `[vocab, H]`
via `Vh = svd(W).Vh` (right singular directions, descending σ). Keep the **centered "bulk" slice**
of `K = H // tau` directions (`left=(H-K)//2`, `right=left+K`), set `basis = Vh[left:right].T`
`[H,K]`. Apply to a pooled embedding as `z = pooled @ basis`, then L2-normalize. This is a
**dimensionality-reduction postprocessor**, NOT a new trained model.

## Why it competes with prefix Matryoshka

The repo already reduces vector size with **prefix Matryoshka** (`matryoshka.truncate_normalized`:
keep the first `d` dims, renormalize) and measured that 256-d retains ~97% of full GermanQuAD nDCG.
EmbedFilter is an *alternative* reduction to the same target dims. It only matters if, at equal
dim, it **beats or matches prefix Matryoshka** while helping active RAG recall — that is exactly
what `scripts/eval_embed_filter_sweep.py` measures (`full` vs `prefix` vs `embedfilter`).

## Important caveat (alignment)

The basis here is the **base** model's unembedding spectrum (`Boldt/Boldt-DC-350M`), but it is
applied to the **fine-tuned** dense embedder's pooled output (the v6.1 checkpoint). The fine-tuned
encoder's representation space may have rotated away from the base unembedding directions, so the
sweep *also* implicitly tests that alignment. The paper applies the lens to a model's own
embeddings. Treat a negative result as a real, honest finding — not a failure to hide.

## Artifacts & honesty rules

- Bases: `scripts/build_embed_filter.py --model Boldt/Boldt-DC-350M --tau {1,2,4,8,16} --out
  outputs/embedfilter/boldt-dc-350m_tau{tau}` → `basis.pt` (git-ignored) + `metadata.json`.
- Sweep: `scripts/eval_embed_filter_sweep.py` → `outputs/v7-embedfilter/sweep.json` + `.md` + run
  card. `--dry-run` imports no ML.
- Gate: `scripts/check_embedfilter_gate.py` (advisory; `--require-real` fails on missing real
  metrics; never fabricates).
- **No number may be claimed unless it is in a saved `outputs/v7-embedfilter/sweep.json` produced
  by a real run.** `local_rag` has no eval set on disk yet and is skipped (not fabricated).

## Advisory gate (not blocking)

- τ=2 / 512-d passes only if mean nDCG@10 and Recall@100 are within 0.005 absolute of full 1024 on
  the **active** eval sets, or better.
- τ=4 / 256-d passes only if it beats or matches **prefix-256** on mean nDCG@10 and Recall@100.
- GermanQuAD / DT-test are near-ceiling guardrails: no **active** regression worse than −0.005
  unless explicitly diagnostic. WebFAQ / local RAG are primary for product RAG recall. GerDaLIR is
  diagnostic-only unless product scope changes.

## Results — first real run (2026-06-22, RTX A6000)

Candidate embedder: dense-v6.1 (`outputs/v6-1-dense-top50/checkpoints/boldt-dense-rag-v6-1`).
EmbedFilter basis = SVD bulk slice of the BASE `Boldt/Boldt-DC-350M` unembedding. Artifacts:
`outputs/v7-embedfilter/sweep.json` (+`.md`), `outputs/v7-embedfilter/gate.json`,
`outputs/v7-embedfilter/unembedding_lens.json`.

**Advisory gate: FAIL.** Conclusion: **EmbedFilter does NOT beat prefix Matryoshka for this German
dense retriever — keep prefix Matryoshka.**

nDCG@10, EmbedFilter (EF) vs prefix at equal dim, active sets:

| dim | webfaq EF / prefix | germanquad EF / prefix | dt_test EF / prefix |
|---|---|---|---|
| full 1024 | 0.7046 | 0.8778 | 0.9748 |
| 512 (τ2) | **0.7050** / 0.7039 | **0.8792** / 0.8732 | 0.9717 / 0.9718 |
| 256 (τ4) | 0.6892 / **0.7044** | 0.8578 / 0.8549 | 0.9660 / 0.9667 |
| 128 (τ8) | 0.6720 / 0.6958 | 0.8148 / 0.8199 | 0.9550 / 0.9550 |
| 64 (τ16) | 0.6204 / 0.6772 | 0.7377 / 0.7479 | 0.9053 / 0.9203 |

Gate checks: τ2/512 within-tol of full ✅; τ4/256 vs prefix-256 mean ΔnDCG **−0.0043** ❌ (dragged by
WebFAQ −0.0152); GermanQuAD/DT-test guardrail worst ΔnDCG/full **−0.02** ❌ (GermanQuAD-256, past the
−0.005 tolerance).

- **512-d:** EmbedFilter matches full and marginally beats prefix — viable but no better than prefix.
- **≤256-d:** EmbedFilter is worse than prefix on the primary WebFAQ set and regresses the
  near-ceiling GermanQuAD guardrail; it degrades faster as dims shrink.
- **GerDaLIR (diagnostic, OOD legal):** EmbedFilter *improves* Recall@100 at every dim (512: 0.2026
  vs prefix 0.1781, even > full 0.1848). Interesting, but legal is retired/diagnostic-only — not a
  promotion signal.

Interpretation (the documented alignment caveat): the base-model unembedding bulk slice keeps the
useful directions at 512-d, but at aggressive reductions (≤256-d) it drops directions the
fine-tuned embedder relies on for in-domain retrieval, so prefix truncation of the embedder's own
space reduces better. The unembedding-lens diagnostic shows the top decoded tokens are ~100%
non-content both before and after the filter — consistent with the base unembedding not aligning
to the fine-tuned embedder's representation space.

No production recommendation and no model-card change: prefix Matryoshka remains the reduction
method. EmbedFilter is retained as an experimental postprocessor (and a possible direction only if
a basis is built from the embedder's own space rather than the base unembedding).
