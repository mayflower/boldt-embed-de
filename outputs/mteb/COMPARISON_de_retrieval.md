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
| e5-base | 512 | 0.923 | 0.153 | 0.530 | 0.263 |
| qwen3-0.6b | 512 | 0.911 | 0.180 | 0.541 | 0.272 |
| lfm2.5 | 512 | 0.913 | 0.150 | 0.520 | 0.237 |

## Notes
- **Boldt** (`v6-*`) uses NO query/doc prefix (symmetric), matching its training. Competitors load
  via `mteb.get_model` (e5/Qwen apply their official prompts) or, for LFM2.5, the documented
  `query:` / `document:` prefixes. e5/gte/Qwen/LFM run at **512** tokens; Boldt is shown at both
  **256** (its training length) and **512** (fair comparison).
- `v6-best-round7` = the AutoResearch hill-climb winner; `v6-1-baseline` = the pre-tuning v6.1
  model. They are ~tied on MTEB retrieval (the loop optimized WebFAQ recall@100, not these tasks).
- MIRACL = the **hard-negatives** variant (reduced corpus, the standard affordable MTEB setting),
  not the full ~2M-passage corpus. MLDR encodes long docs truncated to the model's seq length.
- These tasks (legal GerDaLIR, Wikipedia MIRACL, long-doc MLDR) are largely **out of Boldt's
  training distribution** (WebFAQ/FAQ-style RAG). The comparison reflects scope, not a defect.
