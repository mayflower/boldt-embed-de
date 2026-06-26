# Research track (optional) — hybrid / multi-vector ceiling breaker

**Status: optional, NOT the product default.** This track is a contingency for the case where
single-vector dense retrieval **stagnates on MIRACL / general short-doc retrieval after the data +
merge + distill levers are exhausted** (the v8 program). Dense single-vector remains the primary,
promotable product; nothing here may replace a dense promotion, and every mode carries its own gate
and report so the product/experiment boundary stays clean.

## When to activate
Activate only when ALL of the following hold (recorded with artifacts, per ADR-005):
1. `/ar-frontier` / the controller program has run data-balanced + specialist-merge + listwise-KL.
2. The MTEB(deu) aggregate is still below the same-size-peer frontier, and the gap is concentrated
   in **MIRACL** (general Wikipedia ad-hoc) — the structural single-vector weak spot.
3. The frontier report (`scripts/ar_report.py`) shows no remaining headroom from more data/merge.

If instead the gap is in long-doc (GerDaLIR/MLDR), that is an **eval-context** fix (serve at native
length), not a reason to open this track — see the v8 research notes.

## Candidate modes (each its own gate + report)
| mode | idea | why it can break the dense ceiling | cost / risk |
|---|---|---|---|
| `sparse_dense_hybrid` | fuse BM25/SPLADE sparse score with dense (RRF / weighted) | lexical recall complements dense semantics on rare entities/terms | low (no training); a fusion weight to tune |
| `splade_head` | learned sparse head on the encoder | term-level expansion; strong on exact-match retrieval | medium; needs a sparse training objective |
| `bge_m3_style` | joint dense + sparse + multi-vector, scored separately | one model, three retrieval signals | medium; eval complexity |
| `colbert_late_interaction` | per-token vectors + MaxSim | token-level matching recovers what mean-pooling loses (cf. LFM2.5-ColBERT) | higher index cost; separate serving mode |
| `reranked_two_stage` | dense first-stage + the existing Boldt reranker | a strong reranker lifts nDCG without a better first stage | inference latency; we already have the reranker |

References: BGE-M3 (2402.03216), ColBERTv2 (2112.01488), SPLADE (2107.05720), the LFM2.5-ColBERT
retriever (liquid.ai/blog/lfm2-5-retrievers), and the single-vector ceiling discussion (2508.21038).

## Scope of the current PR
**Plan + stubs only — no large new training.** Two stub CLIs are added so the controller/program can
*plan* this track without it ever masquerading as a dense result:
- `scripts/ar_multivector_plan.py` — emits a dry-run plan for a chosen mode (the steps + artifacts it
  would need), reading `configs/autoresearch/hybrid_track.json`. No training.
- `scripts/ar_hybrid_eval.py` — emits a dry-run plan for a hybrid/late-interaction evaluation
  (which retrieval modes to score, on which tasks), fail-closed on missing inputs. No GPU.

Both are dry-run-only stubs: they print a plan and exit. Promoting anything from this track requires
its own (future) gate — the dense frontier gate (`check_mteb_frontier_gate.py`) is for the dense
product and must not be reused to bless a hybrid system as the dense model.
