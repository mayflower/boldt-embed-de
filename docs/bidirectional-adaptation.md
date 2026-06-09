# Bidirectional adaptation (LLM2Vec-style)

Boldt is a causal decoder. For embeddings, bidirectional attention often pools better than
last-token causal attention because every token sees the whole sequence. This document covers
the **bidirectional track**, its verification diagnostic, and MNTP pre-adaptation.
Implemented in `src/boldt_embed/llm2vec_boldt.py` + `scripts/prepare_bidirectional_student.py`.
The **causal track** (`model_causal` / `train.py`) remains the baseline.

## Causal vs bidirectional

| | causal (baseline) | bidirectional (this track) |
|---|---|---|
| attention | lower-triangular (token *t* sees ≤ *t*) | full (every token sees all non-pad) |
| pooling | EOS / last-token | mean / EOS / latent-attention |
| how | as pretrained | patch `_update_causal_mask` to padding-only |

`enable_bidirectional_attention(model)` patches the decoder's `_update_causal_mask` to a
padding-only mask (requires `attn_implementation="eager"`). The patch is shared with the
training module (`train.enable_bidirectional`) so there is one implementation.

## Verification diagnostic (don't just hope)

Patching internal HF attention is fragile, so we **verify** it numerically rather than
assume it worked. `verify_bidirectional_attention`:

1. Forward a sentence `A B C … Z`; record the hidden state at an **early** token A.
2. Change the **last** token Z; forward again; record A again.
3. Compute the L2 delta at A.

- Under **causal** attention, A cannot attend to Z → delta ≈ 0 (`causal_is_masked`).
- Under **bidirectional** attention, A attends to Z → delta > 0.

The verdict `is_bidirectional` requires `delta_bidirectional > eps` **and** greater than the
causal delta. `prepare_bidirectional_student.py` aborts the run if the verdict is false — a
silently-still-causal model would otherwise train the wrong thing.

## MNTP pre-adaptation

`run_mntp_adaptation` runs masked-next-token-prediction (LLM2Vec step 2): mask a fraction of
tokens and train the now-bidirectional model to recover them, adapting representations to use
both-sided context before contrastive training. Llama has no `[MASK]` token, so masked
positions get random ids (a denoising MNTP variant — see `train.mask_tokens`). MNTP texts are
plain German passages (`data/processed/mntp_texts.jsonl`, an **ignored output** — a tiny
fixture lives at `tests/fixtures/mntp_texts.jsonl`).

## Run it

```bash
# Offline plan (no torch): config + text count + planned steps
python scripts/prepare_bidirectional_student.py \
  --texts tests/fixtures/mntp_texts.jsonl --dry-run

# Real run (GPU): enable -> verify -> MNTP -> export a bi-encoder
python scripts/prepare_bidirectional_student.py \
  --base-model Boldt/Boldt-DC-350M \
  --texts data/processed/mntp_texts.jsonl \
  --output outputs/checkpoints/boldt-bi-mntp \
  --steps 1000 --batch-size 8 --max-length 256 --bf16 --gradient-checkpointing
```

The exported checkpoint (model + tokenizer + `bi_encoder_pooling.json`) feeds the modern
embedding trainer (`docs/modern-embedding-training.md`) as the contrastive starting point.

## Pooling

`pool_embeddings(hidden_states, attention_mask, pooling)` supports `mean` and `eos`/
`last_token` and works on torch tensors **or** plain Python lists (so the pooling shape/mean
logic is unit-tested without torch). After truncating Matryoshka prefixes, re-normalize.

## Known risks

- **Patching internal HF methods** (`_update_causal_mask`) can break across `transformers`
  versions; the verification diagnostic is the guard rail — run it after any upgrade.
- Requires `attn_implementation="eager"`; flash/SDPA paths may ignore the custom mask.
- Bidirectional ≠ automatically better: keep the causal baseline and let evaluation decide
  (`docs/hybrid-retrieval-eval.md`). Checkpoints are never committed.
