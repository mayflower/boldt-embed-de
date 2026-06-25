# MTEB(deu) retrieval-core — model comparison

**Metric:** nDCG@10 (MTEB primary), German subset (`deu`), test/dev split per task.
**Harness:** `scripts/run_mteb_retrieval_de.py` · mteb 2.16.0 · commit `c746ff99be260da6214c5b2de4d1ff51c896b1e0`.
Every number is read from a saved `outputs/mteb/<label>/summary.json` (ADR-005); gte excluded —
its custom remote-code architecture raised a CUDA device-side assert in this environment (2 tries).

| Model | seq | GermanQuAD | GerDaLIR-S | MIRACL-hn | MLDR |
|---|---|---|---|---|---|
| v6-1-baseline | 256 | 0.843 | 0.046 | 0.332 | 0.197 |
| v6-best-round7 | 256 | 0.846 | 0.050 | 0.324 | 0.203 |
| v6-best-round7-512 | 512 | 0.866 | 0.085 | 0.326 | 0.221 |
| v6-best-round7-1024 | 1024 | — | 0.138 | — | 0.237 |
| v6-best-round7-2048 | 2048 | — | 0.195 | — | 0.264 |
| v6-slerp-merge | 512 | 0.865 | 0.083 | 0.336 | 0.226 |
| e5-base | 512 | 0.923 | 0.153 | 0.530 | 0.263 |
| qwen3-0.6b | 512 | 0.911 | 0.180 | 0.541 | 0.272 |
| lfm2.5 | 512 | 0.913 | 0.150 | 0.520 | 0.237 |

## Notes
- **Boldt** (`v6-*`) uses NO query/doc prefix (symmetric), matching its training. Competitors load
  via `mteb.get_model` (e5/Qwen apply their official prompts) or, for LFM2.5, the documented
  `query:` / `document:` prefixes.
- `v6-best-round7` = the AutoResearch hill-climb winner; `v6-1-baseline` = the pre-tuning v6.1
  model. They are ~tied on MTEB retrieval (the loop optimized WebFAQ recall@100, not these tasks).
- MIRACL = the **hard-negatives** variant (reduced corpus, the standard affordable MTEB setting).

## ⚠️ Sequence-length is a major confound on the long-doc tasks (corrected finding)
Boldt was first run at **256** tokens (its training length / an OOM guard), competitors at **512**.
Re-running Boldt up its base's native RoPE context (Llama, 2048) shows the long-doc "gap" was
largely an **eval-truncation artifact**, not model quality:

| round-7 seq | GerDaLIR-S | MLDR |
|---|---|---|
| 256  | 0.050 | 0.203 |
| 512  | 0.085 | 0.221 |
| 1024 | 0.138 | 0.237 |
| **2048** | **0.195** | **0.264** |

At its native 2048 context Boldt **leads GerDaLIR** (0.195 > Qwen 0.180 > e5 0.153 > LFM 0.150) and
**matches e5 on MLDR** (0.264 vs 0.263). **Fairness caveat:** e5/LFM2.5 max out at ~512 tokens, so
their 512 numbers are their ceiling and the GerDaLIR/MLDR comparison vs them is fair; **Qwen3 (32k)
and gte (8192) were capped at 512 here and would likely rise too** — they need a native-context
re-run before any lead is claimed over them on long-doc.

## What is and isn't a real gap
- **Short-doc tasks are unaffected by context** and the gap there is real: GermanQuAD
  (Boldt ~0.866 vs ~0.92) and **MIRACL** (Boldt ~0.33 vs ~0.52) — genuine model-quality/data gaps
  (Wikipedia ad-hoc + general retrieval are out of Boldt's WebFAQ/RAG training distribution).
- **Long-doc tasks (GerDaLIR/MLDR) were mostly an eval-cap artifact** — fixed for free by serving
  at native context. The 256 cap belongs to *training* (memory), not eval/serving of long docs.
- `v6-slerp-merge` (SLERP of round-7 ⊕ v6.1) is a **logged negative**: the two parents are
  behaviourally near-identical so the merge is ~a no-op (noise-level ±). SLERP needs *complementary*
  strong checkpoints; it is a polish step, not a fix for a structural gap.
