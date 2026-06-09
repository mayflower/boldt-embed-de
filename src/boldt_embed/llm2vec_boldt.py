"""LLM2Vec-style bidirectional adapter for the Boldt student (lazy ML imports).

Hardens the bidirectional proof-of-concept into a testable adapter:

* enable + **verify** that attention is truly bidirectional (a numeric diagnostic, not a
  hope),
* MNTP pre-adaptation,
* pooling + export to a SentenceTransformer-compatible bi-encoder.

The pooling math and the verification *delta* computation are pure-stdlib helpers (testable
with fake vectors); model loading, attention patching, MNTP, and forward passes are ML-only
and lazy-imported. The causal path (`model_causal` / `train.py`) remains the baseline.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

# Reuse the validated attention patch + MNTP masking from the training module.
from .train import enable_bidirectional, mask_tokens  # noqa: F401


# ------------------------------------------------------------- stdlib: pooling/delta
def l2_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two vectors. Used by the bidirectional diagnostic."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def masked_mean_pool(hidden: Sequence[Sequence[float]], mask: Sequence[int]) -> List[float]:
    """Mean over non-masked positions. `hidden`: [seq][dim], `mask`: [seq] of 0/1."""
    dim = len(hidden[0]) if hidden else 0
    acc = [0.0] * dim
    n = 0
    for vec, m in zip(hidden, mask):
        if m:
            for i in range(dim):
                acc[i] += vec[i]
            n += 1
    n = max(n, 1)
    return [x / n for x in acc]


def last_token_pool(hidden: Sequence[Sequence[float]], mask: Sequence[int]) -> List[float]:
    """Hidden state of the last non-masked position (EOS/last-token pooling)."""
    last = -1
    for i, m in enumerate(mask):
        if m:
            last = i
    return list(hidden[last]) if last >= 0 else [0.0] * (len(hidden[0]) if hidden else 0)


def _is_torch_tensor(x: Any) -> bool:
    return hasattr(x, "dim") and hasattr(x, "shape") and not isinstance(x, (list, tuple))


def pool_embeddings(hidden_states: Any, attention_mask: Any, pooling: str = "mean") -> Any:
    """Pool token hidden states to one vector per example.

    Works on torch tensors (`hidden_states`: [B,T,H], `mask`: [B,T]) *or* on nested Python
    lists (same shapes) so the pooling shape logic is unit-testable without torch. Returns a
    torch tensor [B,H] for tensor input, or a list[B][H] for list input."""
    if _is_torch_tensor(hidden_states):
        import torch
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # [B,T,1]
        if pooling == "mean":
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            return summed / counts
        if pooling in ("eos", "last_token", "eos_or_last_token"):
            lengths = attention_mask.sum(dim=1).long() - 1
            idx = lengths.clamp(min=0)
            return hidden_states[torch.arange(hidden_states.size(0)), idx]
        raise ValueError(f"unknown pooling '{pooling}'")
    # pure-python path (lists)
    fn = masked_mean_pool if pooling == "mean" else last_token_pool
    return [fn(h, m) for h, m in zip(hidden_states, attention_mask)]


def pooled_output_shape(batch: int, hidden: int) -> tuple:
    return (batch, hidden)


# --------------------------------------------------------------- ML layer (lazy import)
def load_boldt_for_bidirectional(model_name: str, device: Optional[str] = None,
                                 dtype: str = "bfloat16"):
    """Load Boldt with eager attention (required for the bidirectional mask patch).
    Returns (model, tokenizer). ML-only."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}.get(dtype, torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype,
                                      attn_implementation="eager", trust_remote_code=True)
    if device:
        model = model.to(device)
    return model, tok


def enable_bidirectional_attention(model):
    """Public name for the attention patch (delegates to the validated `enable_bidirectional`)."""
    return enable_bidirectional(model)


def _token_delta_when_last_changes(model, tok, text: str, probe_index: int = 1,
                                   device: Optional[str] = None) -> float:
    """Forward `text`; record the hidden state at `probe_index` (an early token A). Change the
    LAST token, forward again, and return the L2 change at A. Under causal attention A cannot
    see the last token (delta ~ 0); under bidirectional attention it can (delta > 0)."""
    import torch

    enc = tok(text, return_tensors="pt")
    if device:
        enc = {k: v.to(device) for k, v in enc.items()}
    ids = enc["input_ids"]
    with torch.no_grad():
        h1 = model(**enc).last_hidden_state[0, probe_index]
        ids2 = ids.clone()
        # replace the last real token with a different id (avoid pad/eos collisions)
        new_id = int((ids2[0, -1].item() + 7) % model.config.vocab_size)
        ids2[0, -1] = new_id
        enc2 = dict(enc)
        enc2["input_ids"] = ids2
        h2 = model(**enc2).last_hidden_state[0, probe_index]
    return float(torch.linalg.vector_norm((h1 - h2).float()).item())


def verify_bidirectional_attention(model, tok, text: str = "Das Haus am See ist sehr groß",
                                   probe_index: int = 1, device: Optional[str] = None,
                                   eps: float = 1e-4) -> Dict[str, Any]:
    """Diagnostic: measure the change-the-last-token delta at an early token under causal vs
    bidirectional attention. Returns deltas + a boolean verdict. ML-only.

    Note: this mutates `model`'s attention to bidirectional (it patches in place). Reload for
    a fresh causal model if needed."""
    delta_causal = _token_delta_when_last_changes(model, tok, text, probe_index, device)
    enable_bidirectional_attention(model)
    delta_bi = _token_delta_when_last_changes(model, tok, text, probe_index, device)
    return {
        "delta_causal": delta_causal,
        "delta_bidirectional": delta_bi,
        "is_bidirectional": delta_bi > eps and delta_bi > delta_causal,
        "causal_is_masked": delta_causal <= eps,
        "probe_index": probe_index,
        "text": text,
    }


def run_mntp_adaptation(model, tok, texts: Sequence[str], steps: int = 100,
                        batch_size: int = 8, max_length: int = 256, lr: float = 5e-5,
                        mask_prob: float = 0.2, device: Optional[str] = None) -> Dict[str, Any]:
    """Masked-next-token-prediction pre-adaptation (LLM2Vec step 2). Trains the bidirectional
    model to predict masked tokens, adapting it to bidirectional context. ML-only."""
    import torch
    from transformers import AutoModelForMaskedLM  # noqa: F401  (documented; we use LM head below)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    special = [getattr(tok, a + "_token_id", None) for a in ("pad", "cls", "sep", "bos", "eos")]
    losses: List[float] = []
    lm_head = getattr(model, "get_output_embeddings", lambda: None)()
    step = 0
    while step < steps:
        for start in range(0, len(texts), batch_size):
            if step >= steps:
                break
            batch = texts[start:start + batch_size]
            enc = tok(list(batch), return_tensors="pt", padding=True, truncation=True,
                      max_length=max_length)
            if device:
                enc = {k: v.to(device) for k, v in enc.items()}
            masked_input, labels, _ = mask_tokens(enc["input_ids"], enc["attention_mask"],
                                                   mask_prob, model.config.vocab_size, special)
            out = model(input_ids=masked_input, attention_mask=enc["attention_mask"])
            hidden = out.last_hidden_state
            if lm_head is None:
                raise RuntimeError("Base model exposes no LM head for MNTP; load with a head.")
            logits = lm_head(hidden)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            step += 1
    return {"status": "ok", "steps": step, "final_loss": losses[-1] if losses else None,
            "mean_loss": sum(losses) / len(losses) if losses else None}


def export_bi_encoder(model, tok, output_dir: str, pooling: str = "mean") -> str:
    """Save the adapted model + tokenizer and a pooling marker so it can be reloaded as a
    bi-encoder. ML-only."""
    import json as _json
    import pathlib

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "bi_encoder_pooling.json").write_text(
        _json.dumps({"pooling": pooling, "bidirectional": True}), encoding="utf-8")
    return str(out)
