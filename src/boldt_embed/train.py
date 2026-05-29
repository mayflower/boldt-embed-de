"""REAL causal-embedder training on GPU (torch + transformers).

This is not a dry-run: it loads the base weights, runs forward/pool/contrastive/backward
on the configured device, and saves a fine-tuned checkpoint. Imported lazily by the
training script so the stdlib gates remain dependency-free.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

# torch / transformers are imported inside functions so this module can be imported
# (for introspection) without the extras present.


def pick_device(device: Optional[str] = None, index: int = 0) -> str:
    import torch

    if device:
        return device
    if torch.cuda.is_available():
        return f"cuda:{index}"
    return "cpu"


def _last_token_pool(last_hidden, attention_mask):
    import torch

    lengths = attention_mask.sum(dim=1) - 1
    idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
    return last_hidden[idx, lengths]


def _mean_pool(last_hidden, attention_mask):
    m = attention_mask.unsqueeze(-1).type_as(last_hidden)
    return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def _pool(strategy: str, last_hidden, attention_mask):
    if strategy == "mean":
        return _mean_pool(last_hidden, attention_mask)
    return _last_token_pool(last_hidden, attention_mask)  # eos / last_token


def info_nce(q, p, hard=None, temperature: float = 0.03):
    """InfoNCE with in-batch negatives + optional per-row hard negatives.

    q, p: [B, H] (L2-normalized). hard: [B, K, H] or None.
    """
    import torch
    import torch.nn.functional as F

    logits = (q @ p.t()) / temperature  # [B, B]
    if hard is not None and hard.numel() > 0:
        hn = torch.einsum("bh,bkh->bk", q, hard) / temperature  # [B, K]
        logits = torch.cat([logits, hn], dim=1)
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)


def train_causal_real(
    config,
    triples: Sequence[dict],
    *,
    output_dir: str,
    device: Optional[str] = None,
    device_index: int = 0,
    epochs: int = 12,
    lr: float = 2e-5,
    max_len: int = 96,
    log: Callable[[str], None] = print,
) -> Dict[str, object]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    from .instructions import format_query

    dev = pick_device(device, device_index)
    log(f"[train] device={dev}")
    tok = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModel.from_pretrained(
        config.model_name_or_path, trust_remote_code=True, torch_dtype=torch.float32
    ).to(dev)
    hidden = int(model.config.hidden_size)
    pooling = "mean" if config.pooling == "mean" else "eos"
    log(f"[train] loaded {config.model_name_or_path} hidden_size={hidden} pooling={pooling}")

    queries = [format_query(config.query_instruction, t["query"]) for t in triples]
    positives = [t["positive"] for t in triples]
    negatives = [(t.get("negatives") or [None])[0] for t in triples]
    has_neg = all(n is not None for n in negatives)

    def embed(texts: List[str]):
        # Append EOS for last-token pooling (E5-style): the pooled token is a true EOS.
        prepared = [t + tok.eos_token for t in texts] if pooling == "eos" else list(texts)
        batch = tok(prepared, padding=True, truncation=True, max_length=max_len,
                    return_tensors="pt").to(dev)
        out = model(**batch).last_hidden_state
        pooled = _pool(pooling, out, batch["attention_mask"])
        return F.normalize(pooled, p=2, dim=1)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses: List[float] = []
    t0 = time.time()
    for epoch in range(epochs):
        opt.zero_grad()
        q = embed(queries)
        p = embed(positives)
        hard = embed(negatives).unsqueeze(1) if has_neg else None  # [B,1,H]
        loss = info_nce(q, p, hard=hard, temperature=config.temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
        log(f"[train] epoch {epoch + 1}/{epochs} loss={losses[-1]:.4f}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(out_path)
    tok.save_pretrained(out_path)

    return {
        "status": "trained",
        "base_model": config.model_name_or_path,
        "device": dev,
        "gpu_name": (torch.cuda.get_device_name(torch.device(dev))
                     if dev.startswith("cuda") else "cpu"),
        "hidden_size": hidden,
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "pooling": pooling,
        "epochs": epochs,
        "lr": lr,
        "temperature": config.temperature,
        "num_triples": len(triples),
        "used_hard_negatives": has_neg,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "loss_curve": losses,
        "wall_time_sec": round(time.time() - t0, 2),
        "checkpoint": str(out_path),
    }


def encode_texts(
    model_path: str,
    texts: Sequence[str],
    *,
    pooling: str = "eos",
    device: Optional[str] = None,
    device_index: int = 0,
    max_len: int = 96,
    batch_size: int = 16,
    append_eos: bool = True,
):
    """Encode texts with a (trained or base) model. Returns a list of python float vectors."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    dev = pick_device(device, device_index)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                      torch_dtype=torch.float32).to(dev).eval()
    add_eos = append_eos and pooling == "eos"
    vectors: List[List[float]] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = [t + tok.eos_token for t in texts[i : i + batch_size]] if add_eos \
                else list(texts[i : i + batch_size])
            batch = tok(chunk, padding=True, truncation=True, max_length=max_len,
                        return_tensors="pt").to(dev)
            out = model(**batch).last_hidden_state
            pooled = _pool(pooling, out, batch["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            vectors.extend(pooled.cpu().tolist())
    return vectors
