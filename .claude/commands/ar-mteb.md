---
description: Run MTEB(deu) retrieval-core on a model + the same-size-peer frontier gate (the promotion eval)
argument-hint: "<model-path-or-label> [label]"
allowed-tools: Bash(conda run *) Bash(tail *) Bash(cat *) Read
disable-model-invocation: true
---
# AutoResearch — MTEB(deu) eval + frontier gate

Evaluate a model on the official MTEB(deu) retrieval-core and judge it against the same-size peers.
Parse `$ARGUMENTS`: first token = model checkpoint path (e.g. `outputs/v8/diverse-causal/checkpoint`);
second (optional) = `label` for `outputs/mteb/<label>/` (default: basename of the path).

This is the PROMOTION eval — separate from the in-loop proxy. Real GPU; runs on the A6000.

1. Run the four retrieval-core tasks (German subset, @512), prefix-free (our model is symmetric):
   ```bash
   CUDA_VISIBLE_DEVICES=0 conda run -n boldtembed python scripts/run_mteb_retrieval_de.py \
     --model <model> --label <label> \
     --tasks GermanQuAD-Retrieval,GerDaLIRSmall,MIRACLRetrievalHardNegatives,MultiLongDocRetrieval \
     --loader st --batch-size 32 --max-seq-length 512
   ```
   (For a BIDIRECTIONAL model add `--bidirectional`; for a registry competitor use `--loader mteb`.)
2. Apply the frontier gate (beats e5-base/LFM2.5 aggregate AND no regression vs the @512 baseline):
   ```bash
   conda run -n boldtembed python scripts/check_mteb_frontier_gate.py \
     --candidate <label> --peers e5-base,lfm2.5 --baseline v6-1-baseline-512 --format markdown
   ```
   (If `v6-1-baseline-512` is absent, first create it: `/ar-mteb outputs/v6-1-dense-top50/checkpoints/boldt-dense-rag-v6-1 v6-1-baseline-512`.)
3. Report: the four nDCG@10 scores, the per-task `beats peer` flags, `tasks_beating_peers`, the
   aggregate vs peer-frontier, and `promotable`. **Never claim a benchmark beyond the saved
   summary** (`outputs/mteb/<label>/summary.json`, ADR-005). MLDR is encoded at 256-token training
   length; note that caveat. Promotion still needs human review.
