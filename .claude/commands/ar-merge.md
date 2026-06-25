---
description: Merge (soup/SLERP) complementary checkpoints, then MTEB-eval + frontier-gate the merge
argument-hint: "<ckptA,ckptB,...> [mean|slerp] [t|weights]"
allowed-tools: Bash(conda run *) Bash(tail *) Bash(cat *) Read
disable-model-invocation: true
---
# AutoResearch — specialist merge (the trade-off-escape lever)

Merge complementary checkpoints into one model that inherits each one's strong task, then eval+gate.
Parse `$ARGUMENTS`: first = comma-separated checkpoint dirs; second = `mean` (default, N-model soup)
or `slerp` (2 models only); third = SLERP `t` (default 0.5) or mean `weights` (comma list).

Merging only helps when the inputs share a BASIN (train specialists from a common warm-start) and
are genuinely COMPLEMENTARY (different strong tasks). Merging near-identical or divergent checkpoints
is a no-op / averages to mediocrity — judge by the gate, not hope.

1. Build the merge (CPU, fast):
   ```bash
   conda run -n boldtembed python scripts/model_soup.py \
     --models <ckptA,ckptB,...> --method <mean|slerp> [--t <t> | --weights <w1,w2,...>] \
     --out outputs/merged/<name>/checkpoint
   ```
2. Eval + gate the merge via `/ar-mteb`:
   ```bash
   CUDA_VISIBLE_DEVICES=0 conda run -n boldtembed python scripts/run_mteb_retrieval_de.py \
     --model outputs/merged/<name>/checkpoint --label <name> \
     --tasks GermanQuAD-Retrieval,GerDaLIRSmall,MIRACLRetrievalHardNegatives,MultiLongDocRetrieval \
     --loader st --batch-size 32 --max-seq-length 512
   conda run -n boldtembed python scripts/check_mteb_frontier_gate.py --candidate <name> \
     --peers e5-base,lfm2.5 --baseline v6-1-baseline-512 --format markdown
   ```
3. Report each task score for the merge AND its parents, the frontier verdict, and — the key
   question — whether the merge keeps ≥~90% of EACH parent's strong task simultaneously (escapes the
   trade-off) or averages down between them. If a t-sweep is requested, run `--t 0.3/0.5/0.7` and
   report the frontier of merges. Merged weights are git-ignored (regenerable).
