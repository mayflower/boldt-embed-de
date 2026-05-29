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


def enable_bidirectional(model):
    """LLM2Vec step 1: replace the causal mask with a padding-only (bidirectional) mask.

    Patches the inner decoder's ``_update_causal_mask`` so every position attends to every
    non-pad position (no causal triangle). Requires the model be loaded with
    ``attn_implementation='eager'``. Returns the same model for chaining.
    """
    import torch

    base = model.model if hasattr(model, "model") else model

    def _bidirectional_mask(self, attention_mask, input_tensor, *args, **kwargs):
        dtype = input_tensor.dtype
        b, t = input_tensor.shape[0], input_tensor.shape[1]
        if attention_mask is None:
            return None  # no padding -> full bidirectional attention
        min_val = torch.finfo(dtype).min
        keep = attention_mask[:, None, None, :].to(dtype)  # [B,1,1,T]
        additive = (1.0 - keep) * min_val                  # 0 where keep, min where pad
        return additive.expand(b, 1, t, t).contiguous()    # [B,1,T,T], no causal triangle

    if hasattr(base, "_update_causal_mask"):
        base._update_causal_mask = _bidirectional_mask.__get__(base, base.__class__)
    if hasattr(base, "config"):
        base.config.is_causal = False
        if hasattr(base.config, "is_decoder"):
            base.config.is_decoder = False
    return model


def mask_tokens(input_ids, attention_mask, mask_prob, vocab_size, special_ids):
    """MNTP masking: return (masked_input, labels, masked_bool).

    labels = -100 except at masked positions (where it holds the original id). Llama has no
    [MASK] token, so masked positions are replaced with random ids (a denoising MNTP variant).
    Special/pad positions are never masked.
    """
    import torch

    labels = input_ids.clone()
    probs = torch.full(input_ids.shape, float(mask_prob))
    special = (attention_mask == 0)
    for sid in special_ids:
        if sid is not None:
            special = special | (input_ids == sid)
    probs.masked_fill_(special, 0.0)
    masked = torch.bernoulli(probs).bool()
    labels[~masked] = -100
    rand = torch.randint(0, int(vocab_size), input_ids.shape, dtype=input_ids.dtype)
    masked_input = input_ids.clone()
    masked_input[masked] = rand[masked]
    return masked_input, labels, masked


def train_bidirectional_real(
    config,
    triples,
    mntp_texts,
    *,
    output_dir: str,
    device_index: int = 0,
    mntp_steps: int = 10,
    contrastive_steps: int = 10,
    mask_prob: float = 0.2,
    lr: float = 2e-5,
    max_len: int = 64,
    temperature: float = 0.03,
    log: Callable[[str], None] = print,
) -> Dict[str, object]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = pick_device(None, device_index)
    log(f"[bi] device={dev}")
    tok = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path, trust_remote_code=True,
        torch_dtype=torch.float32, attn_implementation="eager",
    ).to(dev)
    enable_bidirectional(model)
    hidden = int(model.config.hidden_size)
    special_ids = [tok.pad_token_id, tok.bos_token_id, tok.eos_token_id]
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    enc = tok(list(mntp_texts), padding=True, truncation=True, max_length=max_len,
              return_tensors="pt").to(dev)
    mntp_losses = []
    for step in range(mntp_steps):
        opt.zero_grad()
        masked_input, labels, _ = mask_tokens(
            enc["input_ids"].cpu(), enc["attention_mask"].cpu(), mask_prob,
            model.config.vocab_size, special_ids)
        masked_input = masked_input.to(dev)
        labels = labels.to(dev)
        logits = model(input_ids=masked_input, attention_mask=enc["attention_mask"]).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        mntp_losses.append(float(loss.item()))
        log(f"[bi-mntp] step {step + 1}/{mntp_steps} loss={mntp_losses[-1]:.4f}")

    base = model.model if hasattr(model, "model") else model

    def embed(texts):
        b = tok(list(texts), padding=True, truncation=True, max_length=max_len,
                return_tensors="pt").to(dev)
        h = base(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        return F.normalize(_mean_pool(h, b["attention_mask"]), p=2, dim=1)

    queries = [t["query"] for t in triples]
    positives = [t["positive"] for t in triples]
    negatives = [(t.get("negatives") or [None])[0] for t in triples]
    has_neg = all(n is not None for n in negatives)
    con_losses = []
    for step in range(contrastive_steps):
        opt.zero_grad()
        q = embed(queries)
        p = embed(positives)
        hard = embed(negatives).unsqueeze(1) if has_neg else None
        loss = info_nce(q, p, hard=hard, temperature=temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        con_losses.append(float(loss.item()))
        log(f"[bi-con] step {step + 1}/{contrastive_steps} loss={con_losses[-1]:.4f}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(out_path)
    tok.save_pretrained(out_path)
    return {
        "status": "trained",
        "variant": "bidirectional",
        "base_model": config.model_name_or_path,
        "device": dev,
        "gpu_name": torch.cuda.get_device_name(torch.device(dev)) if dev.startswith("cuda") else "cpu",
        "hidden_size": hidden,
        "adaptation": "bidirectional_attention + MNTP + contrastive",
        "mntp_steps": mntp_steps,
        "contrastive_steps": contrastive_steps,
        "mask_prob": mask_prob,
        "mntp_loss_curve": mntp_losses,
        "contrastive_loss_curve": con_losses,
        "mntp_initial_loss": mntp_losses[0] if mntp_losses else None,
        "mntp_final_loss": mntp_losses[-1] if mntp_losses else None,
        "contrastive_initial_loss": con_losses[0] if con_losses else None,
        "contrastive_final_loss": con_losses[-1] if con_losses else None,
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


def _rerank_inputs(template: str, query: str, documents: Sequence[str]) -> List[str]:
    return [template.replace("{query}", query).replace("{document}", d) for d in documents]


def train_reranker_real(
    config,
    triples,
    *,
    output_dir: str,
    device_index: int = 0,
    epochs: int = 15,
    lr: float = 2e-5,
    max_len: int = 128,
    log: Callable[[str], None] = print,
) -> Dict[str, object]:
    """Train a real cross-encoder reranker (LlamaForSequenceClassification, 1 logit, BCE)."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = pick_device(None, device_index)
    log(f"[rr] device={dev}")
    tok = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path, num_labels=1, trust_remote_code=True,
        torch_dtype=torch.float32).to(dev)
    model.config.pad_token_id = tok.pad_token_id

    texts: List[str] = []
    labels: List[float] = []
    for t in triples:
        texts += _rerank_inputs(config.input_template, t["query"], [t["positive"]])
        labels.append(1.0)
        for neg in t.get("negatives", []) or []:
            texts += _rerank_inputs(config.input_template, t["query"], [neg])
            labels.append(0.0)
    y = torch.tensor(labels, device=dev)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses: List[float] = []
    t0 = time.time()
    for epoch in range(epochs):
        opt.zero_grad()
        enc = tok(texts, padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(dev)
        logits = model(**enc).logits.squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
        log(f"[rr] epoch {epoch + 1}/{epochs} loss={losses[-1]:.4f}")

    # pairwise accuracy: does score(query, positive) > score(query, negative)?
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for t in triples:
            pos = _rerank_inputs(config.input_template, t["query"], [t["positive"]])
            negs = _rerank_inputs(config.input_template, t["query"], t.get("negatives") or [])
            if not negs:
                continue
            enc = tok(pos + negs, padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(dev)
            s = model(**enc).logits.squeeze(-1)
            pos_s, neg_s = s[0], s[1:]
            correct += int((pos_s > neg_s).all().item())
            total += 1
    pairwise_acc = correct / total if total else None

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    tok.save_pretrained(out_path)
    return {
        "status": "trained",
        "variant": "reranker",
        "base_model": config.model_name_or_path,
        "device": dev,
        "gpu_name": torch.cuda.get_device_name(torch.device(dev)) if dev.startswith("cuda") else "cpu",
        "num_positive_pairs": int(sum(labels)),
        "num_negative_pairs": int(len(labels) - sum(labels)),
        "epochs": epochs,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "loss_curve": losses,
        "train_pairwise_accuracy": pairwise_acc,
        "wall_time_sec": round(time.time() - t0, 2),
        "checkpoint": str(out_path),
    }


def rerank_scores_real(
    model_path: str,
    query: str,
    documents: Sequence[str],
    template: str,
    *,
    device_index: int = 0,
    max_len: int = 128,
) -> List[float]:
    """Score (query, document) pairs with a trained reranker -> sigmoid relevance in [0,1]."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = pick_device(None, device_index)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=1, trust_remote_code=True, torch_dtype=torch.float32).to(dev).eval()
    model.config.pad_token_id = tok.pad_token_id
    inputs = _rerank_inputs(template, query, documents)
    with torch.no_grad():
        enc = tok(inputs, padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(dev)
        logits = model(**enc).logits.squeeze(-1)
        return torch.sigmoid(logits).cpu().tolist()


# =============================================================================
# Full-scale (real-data) causal embedder training + GPU retrieval evaluation.
# =============================================================================
def _embed_for_train(model, tok, texts, pooling, max_len, dev):
    import torch.nn.functional as F

    prepared = [t + tok.eos_token for t in texts] if pooling == "eos" else list(texts)
    batch = tok(prepared, padding=True, truncation=True, max_length=max_len,
                return_tensors="pt").to(dev)
    out = model(**batch).last_hidden_state
    return F.normalize(_pool(pooling, out, batch["attention_mask"]), p=2, dim=1)


def train_pairs_real(
    config,
    pairs,
    *,
    output_dir: str,
    device_index: int = 0,
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_len: int = 192,
    pooling: str = "eos",
    temperature: float = 0.05,
    use_amp: bool = True,
    grad_checkpoint: bool = True,
    seed: int = 0,
    max_steps: Optional[int] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, object]:
    """Real MNRL fine-tune over (query, positive) pairs with in-batch negatives.

    Minibatched, multi-epoch, bf16 autocast, optional gradient checkpointing (keeps GPU
    memory low so it fits alongside other jobs). ``pairs`` is a list of (query, positive).
    """
    import random
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = pick_device(None, device_index)
    log(f"[full] device={dev} pairs={len(pairs)} epochs={epochs} bs={batch_size} "
        f"grad_ckpt={grad_checkpoint}")
    tok = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(config.model_name_or_path, trust_remote_code=True,
                                      torch_dtype=torch.float32).to(dev)
    model.train()
    if grad_checkpoint:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = len(pairs)
    losses: List[float] = []
    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        order = list(range(n))
        random.Random(seed + epoch).shuffle(order)
        for b in range(0, n, batch_size):
            chunk = order[b: b + batch_size]
            if len(chunk) < 2:
                continue
            q = [pairs[i][0] for i in chunk]
            p = [pairs[i][1] for i in chunk]
            opt.zero_grad()
            if use_amp and dev.startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    qe = _embed_for_train(model, tok, q, pooling, max_len, dev)
                    pe = _embed_for_train(model, tok, p, pooling, max_len, dev)
                loss = info_nce(qe.float(), pe.float(), temperature=temperature)
            else:
                qe = _embed_for_train(model, tok, q, pooling, max_len, dev)
                pe = _embed_for_train(model, tok, p, pooling, max_len, dev)
                loss = info_nce(qe, pe, temperature=temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
            step += 1
            if step % 25 == 0:
                log(f"[full] epoch {epoch + 1}/{epochs} step {step} loss={losses[-1]:.4f}")
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(out_path)
    tok.save_pretrained(out_path)
    return {
        "status": "trained",
        "base_model": config.model_name_or_path,
        "device": dev,
        "gpu_name": torch.cuda.get_device_name(torch.device(dev)) if dev.startswith("cuda") else "cpu",
        "hidden_size": int(model.config.hidden_size),
        "num_pairs": n,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "pooling": pooling,
        "steps": step,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "mean_last_50_loss": (sum(losses[-50:]) / len(losses[-50:])) if losses else None,
        "wall_time_sec": round(time.time() - t0, 1),
        "checkpoint": str(out_path),
    }


def _encode_tensor(model, tok, texts, pooling, max_len, dev, batch_size=64):
    from contextlib import nullcontext

    import torch
    import torch.nn.functional as F

    vecs = []
    add_eos = pooling == "eos"
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if dev.startswith("cuda") else nullcontext()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i: i + batch_size]
            prepared = [t + tok.eos_token for t in chunk] if add_eos else list(chunk)
            batch = tok(prepared, padding=True, truncation=True, max_length=max_len,
                        return_tensors="pt").to(dev)
            with ctx:
                out = model(**batch).last_hidden_state
            pooled = _pool(pooling, out, batch["attention_mask"]).float()
            vecs.append(F.normalize(pooled, p=2, dim=1))
    return torch.cat(vecs, 0)


def retrieval_eval_real(
    model_path: str,
    corpus,
    queries,
    *,
    pooling: str = "eos",
    device_index: int = 0,
    max_len: int = 192,
    ks=(1, 10, 100),
    log: Callable[[str], None] = print,
) -> Dict[str, float]:
    """Real retrieval eval: encode corpus + queries on GPU, full cosine, ranking metrics.

    corpus: [{"id","text"}]; queries: [{"query","positive_ids"}].
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    from .metrics import aggregate, metrics_for_query

    dev = pick_device(None, device_index)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                      torch_dtype=torch.float32).to(dev).eval()
    corpus_ids = [c["id"] for c in corpus]
    C = _encode_tensor(model, tok, [c["text"] for c in corpus], pooling, max_len, dev)
    Q = _encode_tensor(model, tok, [q["query"] for q in queries], pooling, max_len, dev)
    log(f"[eval] corpus={len(corpus_ids)} queries={len(queries)}")
    sims = Q @ C.t()
    topn = min(max(ks), len(corpus_ids))
    _, topi = torch.topk(sims, topn, dim=1)
    rows = []
    for i, q in enumerate(queries):
        ranked = [corpus_ids[j] for j in topi[i].tolist()]
        rows.append(metrics_for_query(ranked, set(q["positive_ids"]), ks))
    return aggregate(rows)


def mine_hard_negatives_gpu(model_path, query_texts, corpus_texts, query_pos_idx, *,
                            k=1, pooling="eos", device_index=0, max_len=192,
                            query_batch=2048, log=print):
    """Embedding-based (ANCE-style) hard-negative mining on GPU.

    Encodes corpus + queries with the given model, and for each query returns the index of
    its top scoring corpus passage that is NOT its positive. Returns list[int] (one per query).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = pick_device(None, device_index)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                      torch_dtype=torch.float32).to(dev).eval()
    log(f"[mine] encoding corpus={len(corpus_texts)} queries={len(query_texts)}")
    C = _encode_tensor(model, tok, corpus_texts, pooling, max_len, dev, batch_size=128)
    hardneg_idx = []
    for start in range(0, len(query_texts), query_batch):
        qchunk = query_texts[start:start + query_batch]
        Q = _encode_tensor(model, tok, qchunk, pooling, max_len, dev, batch_size=128)
        sims = Q @ C.t()  # [b, N]
        top = torch.topk(sims, k + 1, dim=1).indices.tolist()
        for j, cand in enumerate(top):
            pos = query_pos_idx[start + j]
            choice = next((c for c in cand if c != pos), cand[0])
            hardneg_idx.append(choice)
    del C
    return hardneg_idx


def train_triples_real(
    config,
    triples,
    *,
    output_dir: str,
    device_index: int = 0,
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 2e-5,
    max_len: int = 192,
    pooling: str = "eos",
    temperature: float = 0.05,
    grad_checkpoint: bool = True,
    seed: int = 0,
    init_from: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, object]:
    """MNRL with explicit hard negatives. ``triples`` = list of (query, positive, hardneg|None).
    Candidate pool per step = all in-batch positives + all in-batch hard negatives."""
    import random
    import time as _time

    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    dev = pick_device(None, device_index)
    src = init_from or config.model_name_or_path
    log(f"[triples] device={dev} src={src} triples={len(triples)} bs={batch_size}")
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(src, trust_remote_code=True,
                                      torch_dtype=torch.float32).to(dev)
    model.train()
    if grad_checkpoint:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = len(triples)
    losses = []
    step = 0
    t0 = _time.time()
    for epoch in range(epochs):
        order = list(range(n))
        random.Random(seed + epoch).shuffle(order)
        for b in range(0, n, batch_size):
            chunk = order[b:b + batch_size]
            if len(chunk) < 2:
                continue
            q = [triples[i][0] for i in chunk]
            p = [triples[i][1] for i in chunk]
            negs = [triples[i][2] for i in chunk if len(triples[i]) > 2 and triples[i][2]]
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16) if dev.startswith("cuda") else _nullctx():
                qe = _embed_for_train(model, tok, q, pooling, max_len, dev)
                pe = _embed_for_train(model, tok, p, pooling, max_len, dev)
                ne = _embed_for_train(model, tok, negs, pooling, max_len, dev) if negs else None
            cand = torch.cat([pe, ne], 0) if ne is not None else pe
            logits = (qe.float() @ cand.float().t()) / temperature
            labels = torch.arange(qe.size(0), device=dev)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
            step += 1
            if step % 25 == 0:
                log(f"[triples] epoch {epoch + 1}/{epochs} step {step} loss={losses[-1]:.4f}")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(out_path)
    tok.save_pretrained(out_path)
    return {
        "status": "trained", "init_from": src, "num_triples": n, "with_hard_negatives": True,
        "epochs": epochs, "batch_size": batch_size, "steps": step,
        "initial_loss": losses[0] if losses else None, "final_loss": losses[-1] if losses else None,
        "wall_time_sec": round(_time.time() - t0, 1),
        "gpu_name": torch.cuda.get_device_name(torch.device(dev)) if dev.startswith("cuda") else "cpu",
        "hidden_size": int(model.config.hidden_size), "checkpoint": str(out_path),
    }


def _nullctx():
    from contextlib import nullcontext
    return nullcontext()


def train_reranker_scaled(
    config,
    examples,
    *,
    output_dir: str,
    device_index: int = 0,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_len: int = 256,
    grad_checkpoint: bool = True,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> Dict[str, object]:
    """Minibatched cross-encoder reranker training (BCE) at scale.

    ``examples`` = list of (query, document, label in {0,1}). Unlike the toy
    train_reranker_real (which batches everything at once), this minibatches +
    grad-checkpoints so it scales to tens of thousands of pairs.
    """
    import random
    import time as _time

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = pick_device(None, device_index)
    log(f"[rr-scaled] device={dev} examples={len(examples)} bs={batch_size}")
    tok = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path, num_labels=1, trust_remote_code=True,
        torch_dtype=torch.float32).to(dev)
    model.config.pad_token_id = tok.pad_token_id
    model.train()
    if grad_checkpoint:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = len(examples)
    losses = []
    step = 0
    t0 = _time.time()
    for epoch in range(epochs):
        order = list(range(n))
        random.Random(seed + epoch).shuffle(order)
        for b in range(0, n, batch_size):
            chunk = order[b:b + batch_size]
            texts = [config.input_template.replace("{query}", examples[i][0]).replace(
                "{document}", examples[i][1]) for i in chunk]
            y = torch.tensor([float(examples[i][2]) for i in chunk], device=dev)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16) if dev.startswith("cuda") else _nullctx():
                logits = model(**tok(texts, padding=True, truncation=True, max_length=max_len,
                                     return_tensors="pt").to(dev)).logits.squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
            step += 1
            if step % 50 == 0:
                log(f"[rr-scaled] epoch {epoch + 1}/{epochs} step {step} loss={losses[-1]:.4f}")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(out_path)
    tok.save_pretrained(out_path)
    return {
        "status": "trained", "num_examples": n,
        "num_positive": int(sum(1 for e in examples if e[2] >= 0.5)),
        "num_negative": int(sum(1 for e in examples if e[2] < 0.5)),
        "epochs": epochs, "batch_size": batch_size, "steps": step,
        "initial_loss": losses[0] if losses else None, "final_loss": losses[-1] if losses else None,
        "wall_time_sec": round(_time.time() - t0, 1),
        "gpu_name": torch.cuda.get_device_name(torch.device(dev)) if dev.startswith("cuda") else "cpu",
        "checkpoint": str(out_path),
    }


def rerank_scores_batch(model_path, pairs, *, template, device_index=0, max_len=256, batch_size=64):
    """Score many (query, document) pairs with a trained reranker -> list[float] (sigmoid)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = pick_device(None, device_index)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=1, trust_remote_code=True, torch_dtype=torch.float32).to(dev).eval()
    model.config.pad_token_id = tok.pad_token_id
    out = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i + batch_size]
            texts = [template.replace("{query}", q).replace("{document}", d) for q, d in chunk]
            logits = model(**tok(texts, padding=True, truncation=True, max_length=max_len,
                                 return_tensors="pt").to(dev)).logits.squeeze(-1)
            out.extend(torch.sigmoid(logits).cpu().tolist())
    return out
