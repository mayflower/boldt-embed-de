#!/usr/bin/env python3
"""v8 Stage 0a — Masked Next-Token Prediction (MNTP) adaptation for bidirectional conversion.

LLM2Vec step 2: after flipping the decoder to bidirectional attention, the model must RELEARN to
use future context, or contrastive fine-tuning collapses/under-trains (observed: v8 stage-0 attempt
without MNTP cratered to ~0.005 MTEB). MNTP masks tokens and predicts the masked token at position i
from the representation at i-1, with bidirectional attention — teaching the model to read both ways.

Proper minibatched loop (the repo's train_bidirectional_real is a single-batch PoC). Starts from the
ORIGINAL base (which still has the lm_head MNTP needs) and saves an adapted CausalLM checkpoint that
the contrastive stages then load as a SentenceTransformer with bidirectional=True.

Needs the [train] extra + GPU. Emits a run card.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_texts(path: str) -> list:
    import json as _j
    out = []
    for ln in Path(path).read_text(encoding="utf-8").split("\n"):
        ln = ln.strip()
        if ln:
            r = _j.loads(ln)
            t = r.get("text") or r.get("document") or r.get("query")
            if t:
                out.append(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", default="Boldt/Boldt-DC-350M")
    ap.add_argument("--mntp-texts", default="data/processed/mntp_texts.jsonl")
    ap.add_argument("--output", default="outputs/v8/mntp-base")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--mask-prob", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--run-id", default="v8-mntp")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from boldt_embed.train import enable_bidirectional, mask_tokens

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    texts = _read_texts(str(ROOT / args.mntp_texts) if not Path(args.mntp_texts).is_absolute()
                         else args.mntp_texts)
    if not texts:
        raise SystemExit(f"no MNTP texts found in {args.mntp_texts}")

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager").to(dev)
    enable_bidirectional(model)            # bidirectional mask (verified: train.enable_bidirectional)
    model.gradient_checkpointing_enable()
    model.train()
    special_ids = [tok.pad_token_id, tok.bos_token_id, tok.eos_token_id]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup_ratio * args.steps), args.steps)

    started, losses = time.time(), []
    bs, n = args.batch_size, len(texts)
    # deterministic shuffle without Math.random: rotate the index window each step
    for step in range(args.steps):
        beg = (step * bs) % n
        batch = texts[beg:beg + bs] or texts[:bs]
        enc = tok(batch, padding=True, truncation=True, max_length=args.max_len,
                  return_tensors="pt")
        masked, labels, _ = mask_tokens(enc["input_ids"], enc["attention_mask"],
                                        args.mask_prob, model.config.vocab_size, special_ids)
        masked, labels = masked.to(dev), labels.to(dev)
        attn = enc["attention_mask"].to(dev)
        opt.zero_grad()
        logits = model(input_ids=masked, attention_mask=attn).logits
        # MNTP: predict masked token at i from hidden at i-1 → shift logits left by one
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               labels[:, 1:].reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
        if (step + 1) % 100 == 0 or step == 0:
            print(f"[mntp] step {step+1}/{args.steps} loss={losses[-1]:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e}", flush=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    card = {"run_id": args.run_id, "stage": "v8-0a-mntp", "base_model": args.base_model,
            "steps": args.steps, "batch_size": bs, "max_len": args.max_len,
            "mask_prob": args.mask_prob, "lr": args.lr, "n_texts": n,
            "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
            "loss_mean_last50": round(sum(losses[-50:]) / min(50, len(losses)), 4),
            "elapsed_seconds": round(time.time() - started, 1), "output": str(out),
            "bidirectional": True}
    (out / "mntp_run_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(json.dumps(card, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
