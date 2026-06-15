"""ML measurement + LoRA tuning for the v5 small-model bake-off.

ALL heavy imports (torch/transformers/sentence_transformers/peft) are lazy — done inside
functions, never at module import — so importing this module is side-effect-free and the dry-run
path that never calls these functions stays ML-free. GPU-bound; the exact backend specifics
(esp. Qwen3-Reranker scoring) are refined on the first real run, and per-candidate failures are
captured by the caller rather than aborting the bake-off.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Tuple


def _params_m(model) -> float:
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 1)


def _vram_mb() -> float:
    import torch
    return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if torch.cuda.is_available() else 0.0


def _reset_vram() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ------------------------------------------------------------------ dense retrieval
def _load_dense(candidate: Dict[str, Any], device: str) -> Tuple[Callable, float]:
    backend = candidate["backend"]
    if backend == "sentence_transformers":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(candidate["model_name_or_path"], device=device)

        def encode(texts, instr):
            pref = instr or ""
            return model.encode([pref + t for t in texts],
                                normalize_embeddings=candidate.get("normalize", True),
                                convert_to_numpy=True, batch_size=candidate.get("batch_size", 32))
        return encode, _params_m(model)
    if backend == "boldt_causal":
        from boldt_embed.model_causal import CausalEmbedder  # existing boldt encoder
        emb = CausalEmbedder.from_pretrained(candidate["model_name_or_path"], device=device)

        def encode(texts, instr):
            return emb.encode(texts, instruction=instr,
                              normalize=candidate.get("normalize", True))
        return encode, _params_m(emb.model)
    raise ValueError(f"unknown dense backend: {backend}")


def measure_dense(candidate: Dict[str, Any], eval_set: List[Dict[str, Any]],
                  corpus: List[Dict[str, Any]], device: str) -> Dict[str, Any]:
    import numpy as np

    from boldt_embed.metrics import ndcg_at_k
    _reset_vram()
    encode, params = _load_dense(candidate, device)
    doc_ids = [d["doc_id"] for d in corpus]
    dmat = np.asarray(encode([d.get("text", "") for d in corpus], candidate.get("document_instruction")))
    q_texts = [q["query"] for q in eval_set]
    t0 = time.perf_counter()
    qmat = np.asarray(encode(q_texts, candidate.get("query_instruction")))
    t1 = time.perf_counter()

    def ndcg_at_dim(dim: int) -> float:
        qd, dd = qmat[:, :dim], dmat[:, :dim]
        qd = qd / (np.linalg.norm(qd, axis=1, keepdims=True) + 1e-9)
        dd = dd / (np.linalg.norm(dd, axis=1, keepdims=True) + 1e-9)
        sims = qd @ dd.T
        scores = []
        for i, q in enumerate(eval_set):
            order = [doc_ids[j] for j in np.argsort(-sims[i])[:50]]
            pos = q.get("positive_doc_ids") or ([q["positive_doc_id"]] if q.get("positive_doc_id") else [])
            scores.append(ndcg_at_k(order, set(pos), 10))
        return sum(scores) / len(scores) if scores else 0.0

    full = int(candidate.get("expected_dim") or qmat.shape[1])
    quality = ndcg_at_dim(full)
    ret = (ndcg_at_dim(256) / quality) if quality > 0 and full >= 256 else None
    n = max(len(q_texts), 1)
    return {"name": candidate["name"], "family": candidate["family"],
            "quality": round(quality, 4), "latency_ms": round((t1 - t0) / n * 1000, 3),
            "throughput_qps": round(len(q_texts) / max(t1 - t0, 1e-9), 1),
            "params_m": params, "vram_mb": _vram_mb(),
            "retention_256d": round(ret, 4) if ret is not None else None,
            "storage_bytes_per_vector": {str(d): d * 4 for d in candidate.get("embedding_dims", [])}}


# ------------------------------------------------------------------ reranker
def _load_reranker(candidate: Dict[str, Any], device: str) -> Tuple[Callable, float]:
    backend = candidate["backend"]
    if backend == "boldt_reranker":
        from boldt_embed import reranker_modern as RM
        tmpl = candidate.get("input_template", "Anfrage: {query}\nDokument: {document}")

        def score(pairs):
            return RM.score_with_student_reranker(candidate["model_name_or_path"], pairs, tmpl,
                                                  max_length=candidate.get("max_length", 256))
        return score, None
    if backend == "qwen3_reranker":
        # Qwen3-Reranker: yes/no relevance via a chat-style cross-encoder.
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(candidate["model_name_or_path"], device=device,
                             max_length=candidate.get("max_length", 512))

        def score(pairs):
            return list(model.predict(pairs))
        return score, _params_m(model.model)
    raise ValueError(f"unknown reranker backend: {backend}")


def measure_reranker(candidate: Dict[str, Any], fixed_lists: List[Dict[str, Any]],
                     device: str) -> Dict[str, Any]:
    from boldt_embed import hardness_aware_eval as H
    _reset_vram()
    score, params = _load_reranker(candidate, device)
    t0 = time.perf_counter()
    per_query = []
    for r in fixed_lists:
        cands = r.get("candidates") or []
        scores = score([(r.get("query", ""), c.get("text", "")) for c in cands])
        sc = {c["doc_id"]: float(s) for c, s in zip(cands, scores)}
        m = H.list_metrics(r, reranker_scores=sc)
        if m:
            per_query.append(m)
    t1 = time.perf_counter()
    n = max(len(fixed_lists), 1)
    quality = sum(m["reranked_ndcg@10"] for m in per_query) / max(len(per_query), 1)
    return {"name": candidate["name"], "family": candidate["family"],
            "quality": round(quality, 4), "latency_ms": round((t1 - t0) / n * 1000, 3),
            "throughput_qps": round(len(fixed_lists) / max(t1 - t0, 1e-9), 1),
            "params_m": params, "vram_mb": _vram_mb()}


# ------------------------------------------------------------------ optional LoRA tuning
def lora_tune_candidates(config: Dict[str, Any], args) -> Dict[str, Any]:
    """LoRA-tune the requested Qwen3-0.6B candidate(s) on v5 teacher-scored lists; append the tuned
    variant(s) so they are measured under the SAME harness. LoRA unless ``--full-finetune``."""
    tuning = config.get("tuning", {})
    method = "full" if args.full_finetune else "lora"
    out = dict(config)
    if args.tune_reranker:
        tuned = _tune_one(config, role="reranker", method=method, tuning=tuning, device=args.device)
        out["reranker_candidates"] = config["reranker_candidates"] + [tuned]
    if args.tune_embedding:
        tuned = _tune_one(config, role="embedding", method=method, tuning=tuning, device=args.device)
        out["dense_candidates"] = config["dense_candidates"] + [tuned]
    return out


def _tune_one(config: Dict[str, Any], *, role: str, method: str, tuning: Dict[str, Any],
              device: str) -> Dict[str, Any]:
    # Heavy: transformers + peft + the v5 teacher-scored lists. Implemented lazily; the produced
    # checkpoint path is added as a new candidate of the same family for same-harness comparison.
    from pathlib import Path

    base = ("Qwen/Qwen3-Reranker-0.6B" if role == "reranker" else "Qwen/Qwen3-Embedding-0.6B")
    out_dir = Path("outputs/v5-small-rag") / f"{role}-qwen3-0.6b-{method}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_lora_training(base, role, method, tuning, str(out_dir), device)
    name = f"qwen3-{'rerank' if role == 'reranker' else 'emb'}-0.6b-{method}"
    backend = "qwen3_reranker" if role == "reranker" else "sentence_transformers"
    return {"name": name, "model_name_or_path": str(out_dir), "family": "qwen3-tuned",
            "backend": backend, "max_length": 512, "expected_dim": 1024,
            "embedding_dims": [1024, 512, 256], "normalize": True}


def _run_lora_training(base: str, role: str, method: str, tuning: Dict[str, Any],
                       out_dir: str, device: str) -> None:
    import torch  # noqa: F401
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from boldt_embed import data_pipeline as DP
    rows = list(DP.stream_jsonl(tuning["train_lists"]))
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1).to(device)
    if method == "lora":
        lc = tuning.get("lora", {})
        model = get_peft_model(model, LoraConfig(
            r=lc.get("r", 16), lora_alpha=lc.get("alpha", 32), lora_dropout=lc.get("dropout", 0.05),
            target_modules=lc.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])))
    # Pointwise MSE to teacher_softmax_target over the candidate lists (MarginMSE-compatible).
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4 if method == "lora" else 1e-5)
    model.train()
    for r in rows:
        cands = r.get("candidates") or []
        if not cands:
            continue
        enc = tok([r.get("query", "")] * len(cands), [c.get("text", "") for c in cands],
                  truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
        target = torch.tensor([c.get("teacher_softmax_target", 0.0) for c in cands],
                              device=device).float()
        pred = model(**enc).logits.squeeze(-1)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        opt.step()
        opt.zero_grad()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
