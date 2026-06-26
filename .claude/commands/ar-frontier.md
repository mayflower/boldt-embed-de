---
description: AUTONOMOUS frontier program — agent orchestrates train/specialist/merge/distill to beat the same-size peers on MTEB(deu)
argument-hint: "[rounds] [dry|real]"
allowed-tools: Bash(conda run *) Bash(tail *) Bash(cat *) Read Edit
disable-model-invocation: true
---
# AutoResearch — autonomous frontier program

Run the **whole** merge + train + distill program yourself, back-to-back in THIS turn — no manual
per-step command from me. The fine-grained commands (`/ar-data`, `/ar-specialist`, `/ar-merge`,
`/ar-distill`, `/ar-mteb`) are your PRIMITIVES; here you compose them autonomously and decide each
move from the measured state.

Parse `$ARGUMENTS`: first token = rounds **N** (default 4); second = `dry` (default — PLAN only,
print the moves you WOULD make, no GPU) or `real` (execute on the RTX A6000).

**Objective:** beat the **same-size-peer frontier** (e5-base 278M, LFM2.5 350M) on the **aggregate**
of MTEB(deu) retrieval-core (GermanQuAD / GerDaLIR / MIRACL / MLDR, nDCG@10 @512), without regressing
any task below the @512 baseline. The gate is `scripts/check_mteb_frontier_gate.py` (the in-loop
WebFAQ proxy is a *different*, faster objective driven by `/ar-run` — not this).

Setup (once): skim `AUTORESEARCH.md` §"v8+ frontier program". Read the program state:
```bash
conda run -n boldtembed python scripts/ar_frontier_status.py --format markdown
```
Ensure the @512 baseline exists (the gate's no-regress floor); if `v6-1-baseline-512` is absent,
your first real move is to create it (`/ar-mteb outputs/v6-1-dense-top50/checkpoints/boldt-dense-rag-v6-1 v6-1-baseline-512`).

For round k = 1..N, choose **ONE** move from the menu by reading `ar_frontier_status.py` — pick the
move most likely to raise the frontier aggregate or fill the biggest per-task gap:

- **train-balanced** — a generalist. `/ar-data`-style: edit ONLY `current.json` (`data_mixture` over
  several catalogue sources, `runtime.materialize_mixture=true`, `mixture_total`, `faq_cap`,
  `training.max_steps`), then train: `ar_loop.py --real --allow-gpu --allow-checkpoints --run-id
  fr-balanced-k --out-root outputs/v8/frontier`.
- **train-specialist** — a domain expert for MERGING. `data_mixture={one_source:1.0}` + materialize +
  `runtime.train_base_model=<shared warm-start>` (ALL specialists from the SAME warm-start so they
  share a basin) + `training.max_steps`, then train as above with `--run-id fr-spec-<src>-k`.
- **merge** — the trade-off escape. From `per_task_leaders` / `complementary_merge_inputs` pick
  COMPLEMENTARY checkpoints (different tasks lead) that share a basin:
  `scripts/model_soup.py --models <a,b,..> --method mean --out outputs/v8/frontier/fr-merge-k/checkpoint`
  (or `--method slerp --t 0.5` for exactly 2). No GPU-train; cheap.
- **distill** — sharpen ranking. listwise-KL on the current frontier-best:
  `scripts/train_listwise_kl.py --base <best-ckpt> --lists data/processed/v6/reranker_train_lists_teacher_scored.jsonl --output outputs/v8/frontier/fr-distill-k/checkpoint --steps 1500 --contrastive-weight 0.0 --run-id fr-distill-k`.

Then, for EVERY produced checkpoint (all eval at the SAME @512 so the aggregate is comparable):
```bash
CUDA_VISIBLE_DEVICES=0 conda run -n boldtembed python scripts/run_mteb_retrieval_de.py \
  --model <ckpt> --label <run-id> \
  --tasks GermanQuAD-Retrieval,GerDaLIRSmall,MIRACLRetrievalHardNegatives,MultiLongDocRetrieval \
  --loader st --batch-size 32 --max-seq-length 512
conda run -n boldtembed python scripts/check_mteb_frontier_gate.py --candidate <run-id> \
  --peers e5-base,lfm2.5 --baseline v6-1-baseline-512 --format json
```
After any move that edited `current.json`, run the integrity check; if it flags anything OUTSIDE
`configs/autoresearch/experiments/*.json`, **revert that edit and STOP**:
```bash
conda run -n boldtembed python scripts/check_autoresearch_integrity.py --format json
```

Hill-climb: keep a running registry of every candidate's 4-task scores + aggregate (re-read
`ar_frontier_status.py`). The frontier-best is the highest aggregate; **keep ALL checkpoints** (merge
needs the complementary ones, not just the best). Warm-start future specialists from the current
generalist-best so a later merge stays in-basin.

**Stop early** if: a round's gate is `promotable:true`; OR the aggregate hasn't improved for 2
rounds; OR the integrity check fails; OR `dry` mode (after printing the full planned move sequence).

At the end print: the candidate × move × 4-task × aggregate × beats-peer table, the frontier-best +
its gap to the peer aggregate, and the recommended next move. **Rules:** never claim a benchmark
beyond the saved `outputs/mteb/<label>/summary.json` (ADR-005); only `current.json` is editable
(merge/distill produce new checkpoints, they don't touch protected surfaces); the frontier gate runs
OUTSIDE the loop; leakage stays fail-closed (materialized mixes are scanned-clean by construction);
no committed weights (everything lands under git-ignored `outputs/v8/frontier/`).

GPU/cost note: in `real` mode this trains and evaluates **multiple** models on the A6000 — a training
round is ~5–15 min, each MTEB eval ~10 min. **Tell me the planned move sequence and rough cost before
launching a long real program (N≥3); run `dry` first to show the plan.** Prune intermediate
checkpoints you've already evaluated if disk gets tight.
