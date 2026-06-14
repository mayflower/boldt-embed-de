"""Modern reranker training: pointwise / pairwise / listwise + teacher distillation.

Replaces the BCE-only path with three objectives, all fed from the teacher cache:

* **pointwise** — BCEWithLogits on binary labels (or MSE-regress the teacher score).
* **pairwise** — MarginRankingLoss enforcing positive > negative.
* **listwise** — KL/CE of the student's candidate distribution toward the teacher's
  (softmax of teacher scores) — i.e. distillation.

Layered as before: the example/label *builders* and the *metric* helpers are pure stdlib
(testable with fixtures); model loading, the loss objects, and training loops are ML-only and
lazy-imported. The old `train.train_reranker_scaled` (BCE) stays as a baseline.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .metrics import aggregate, metrics_for_query


# ---------------------------------------------------------------- stdlib: builders
def _group_by_query(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_q: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_q.setdefault(str(r["query_id"]), []).append(r)
    return by_q


def _teacher_score(row: Dict[str, Any]) -> Optional[float]:
    if row.get("reranker_score") is not None:
        return float(row["reranker_score"])
    if row.get("embedding_score") is not None:
        return float(row["embedding_score"])
    return None


def build_reranker_examples_from_teacher_cache(rows: Sequence[Dict[str, Any]],
                                               label_mode: str = "binary"
                                               ) -> List[Dict[str, Any]]:
    """Pointwise examples: {query, document, label}. ``binary`` → 1.0/0.0 from `positive`;
    ``teacher`` → the (sigmoid of the) teacher score as a soft regression target."""
    out = []
    for r in rows:
        if label_mode == "teacher":
            s = _teacher_score(r)
            label = 1.0 / (1.0 + math.exp(-s)) if s is not None else (1.0 if r.get("positive") else 0.0)
        else:
            label = 1.0 if r.get("positive") is True else 0.0
        out.append({"query": r.get("query", ""), "document": r["document"], "label": float(label)})
    return out


def build_pairwise_examples(rows: Sequence[Dict[str, Any]], max_pairs_per_query: int = 8
                            ) -> List[Dict[str, Any]]:
    """Pairwise examples {query, positive, negative} from grouped cache rows (every positive
    crossed with every negative, capped per query). Deterministic order."""
    out = []
    for qid, grp in _group_by_query(rows).items():
        pos = [r for r in grp if r.get("positive") is True]
        neg = [r for r in grp if r.get("positive") is not True]
        if not pos or not neg:
            continue
        query = grp[0].get("query", "")
        made = 0
        for p in pos:
            for n in neg:
                out.append({"query": query, "positive": p["document"], "negative": n["document"]})
                made += 1
                if made >= max_pairs_per_query:
                    break
            if made >= max_pairs_per_query:
                break
    return out


def softmax(scores: Sequence[float], temperature: float = 1.0) -> List[float]:
    if not scores:
        return []
    t = max(temperature, 1e-6)
    m = max(scores)
    exps = [math.exp((s - m) / t) for s in scores]
    z = sum(exps) or 1.0
    return [e / z for e in exps]


def build_listwise_batches(rows: Sequence[Dict[str, Any]], temperature: float = 1.0
                           ) -> List[Dict[str, Any]]:
    """Per-query listwise batches: {query, documents[], teacher_scores[], target[], labels[]}.
    ``target`` is the softmax of teacher scores (the distribution the student distills toward)."""
    out = []
    for qid, grp in _group_by_query(rows).items():
        scored = [(r, _teacher_score(r)) for r in grp]
        if not scored or any(s is None for _, s in scored):
            continue
        docs = [r["document"] for r, _ in scored]
        scores = [s for _, s in scored]
        labels = [1.0 if r.get("positive") is True else 0.0 for r, _ in scored]
        out.append({
            "query": grp[0].get("query", ""),
            "documents": docs,
            "teacher_scores": scores,
            "target": softmax(scores, temperature),
            "labels": labels,
        })
    return out


# ----------------------------------------------- stdlib: candidate-list builders (v2)
def candidate_lists_to_pointwise(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """{query, document, label} from candidate-list rows (skips label=None candidates)."""
    out = []
    for r in rows:
        for c in r.get("candidates", []):
            if c.get("label") is None:
                continue
            out.append({"query": r.get("query", ""), "document": c.get("document", ""),
                        "label": float(c["label"])})
    return out


def candidate_lists_to_pairwise(rows: Sequence[Dict[str, Any]], max_pairs_per_query: int = 8,
                                min_teacher_margin: Optional[float] = None
                                ) -> List[Dict[str, Any]]:
    """{query, positive, negative} pairs from candidate lists (label 1 × label 0).

    With ``min_teacher_margin`` set (v3), a pair is emitted ONLY when both sides carry a teacher
    score AND ``pos_score - neg_score >= margin`` — i.e. pairwise margin is applied only where the
    teacher signal is strong, never on ambiguous pairs."""
    out = []
    for r in rows:
        pos = [c for c in r.get("candidates", []) if c.get("label") == 1]
        neg = [c for c in r.get("candidates", []) if c.get("label") == 0]
        made = 0
        for p in pos:
            for n in neg:
                if min_teacher_margin is not None:
                    ps, ns = p.get("teacher_score"), n.get("teacher_score")
                    if ps is None or ns is None or (float(ps) - float(ns)) < min_teacher_margin:
                        continue
                out.append({"query": r.get("query", ""), "positive": p.get("document", ""),
                            "negative": n.get("document", "")})
                made += 1
                if made >= max_pairs_per_query:
                    break
            if made >= max_pairs_per_query:
                break
    return out


# ------------------------------------------------------------- v3: high-precision labeling
V3_POSITIVE_THRESHOLD = 4.0     # teacher reranker score for a high-precision positive
V3_NEG_MARGIN = 2.0             # a clear negative scores <= positive_threshold - margin


def v3_label(teacher_score: Optional[float], positive_threshold: float = V3_POSITIVE_THRESHOLD,
             neg_margin: float = V3_NEG_MARGIN) -> Optional[int]:
    """High-precision label from a teacher reranker score: 1 (>= threshold), 0 (clearly below by
    margin), or None (UNCERTAIN — used for listwise soft targets, never as a hard BCE negative)."""
    if teacher_score is None:
        return None
    s = float(teacher_score)
    if s >= positive_threshold:
        return 1
    if s <= positive_threshold - neg_margin:
        return 0
    return None


def reranker_training_summary(rows: Sequence[Dict[str, Any]],
                              positive_threshold: float = V3_POSITIVE_THRESHOLD) -> Dict[str, Any]:
    """Pre-training visibility over candidate-list rows: positive/negative teacher-score
    separation per domain, uncertain (label=null) count, candidate-source distribution, and the
    synthetic-vs-real share."""
    import statistics
    pos_by_dom: Dict[str, List[float]] = {}
    neg_by_dom: Dict[str, List[float]] = {}
    uncertain = 0
    src_dist: Dict[str, int] = {}
    syn = real = 0
    n_pos = n_neg = 0
    for r in rows:
        for c in r.get("candidates", []):
            dom = str(c.get("domain") or r.get("domain") or "unknown")
            ts = c.get("teacher_score")
            lab = c.get("label")
            if lab == 1:
                n_pos += 1
                if ts is not None:
                    pos_by_dom.setdefault(dom, []).append(float(ts))
            elif lab == 0:
                n_neg += 1
                if ts is not None:
                    neg_by_dom.setdefault(dom, []).append(float(ts))
            else:
                uncertain += 1
            for s in (c.get("candidate_source") or "unknown").split("+") if isinstance(
                    c.get("candidate_source"), str) else [c.get("candidate_source") or "unknown"]:
                src_dist[s] = src_dist.get(s, 0) + 1
            if c.get("synthetic"):
                syn += 1
            else:
                real += 1
    sep = {}
    for dom in sorted(set(pos_by_dom) | set(neg_by_dom)):
        pm = round(statistics.median(pos_by_dom[dom]), 4) if pos_by_dom.get(dom) else None
        nm = round(statistics.median(neg_by_dom[dom]), 4) if neg_by_dom.get(dom) else None
        sep[dom] = {"pos_median": pm, "neg_median": nm,
                    "separation": round(pm - nm, 4) if (pm is not None and nm is not None) else None}
    total_cand = syn + real
    min_pos = None
    pos_scores = [s for v in pos_by_dom.values() for s in v]
    if pos_scores:
        min_pos = round(min(pos_scores), 4)
    return {
        "positives": n_pos, "negatives": n_neg, "uncertain": uncertain,
        "separation_by_domain": sep,
        "candidate_source_distribution": dict(sorted(src_dist.items())),
        "synthetic_candidates": syn, "real_candidates": real,
        "synthetic_share": round(syn / total_cand, 4) if total_cand else 0.0,
        "positive_threshold": positive_threshold,
        "min_positive_teacher_score": min_pos,
        "high_precision_positives": (min_pos is None or min_pos >= positive_threshold),
    }


def candidate_lists_to_listwise(rows: Sequence[Dict[str, Any]], temperature: float = 1.0
                                ) -> List[Dict[str, Any]]:
    """Per-query listwise batch: target = softmax of teacher_scores when present, else the
    (normalized) labels. {query, documents, target, labels}."""
    out = []
    for r in rows:
        cands = r.get("candidates", [])
        if len(cands) < 2:
            continue
        docs = [c.get("document", "") for c in cands]
        labels = [float(c.get("label") or 0) for c in cands]
        scores = [c.get("teacher_score") for c in cands]
        if all(s is not None for s in scores):
            target = softmax([float(s) for s in scores], temperature)
        else:
            z = sum(labels) or 1.0
            target = [l / z for l in labels]
        out.append({"query": r.get("query", ""), "documents": docs, "target": target,
                    "labels": labels})
    return out


_RERANKER_LOSS_COMPONENTS = {
    "pointwise": ["BCEWithLogitsLoss"],
    "pairwise": ["MarginRankingLoss"],
    "listwise": ["KLDivLoss(listwise)"],
    "mixed": ["BCEWithLogitsLoss", "MarginRankingLoss", "KLDivLoss(listwise)"],
}


def plan_reranker_loss(loss: str) -> Dict[str, Any]:
    """Describe the loss stack (stdlib) for a chosen reranker objective."""
    return {"loss": loss, "components": _RERANKER_LOSS_COMPONENTS.get(loss, [])}


# ---------------------------------------------------------------- stdlib: metrics
def rerank_metrics(candidate_ids: Sequence[str], scores: Sequence[float],
                   positive_ids, ks: Sequence[int] = (10,)) -> Dict[str, float]:
    """Sort candidates by score (desc) and compute retrieval metrics. Stable: ties keep input
    order (so a no-op scorer reproduces the first-stage ranking)."""
    order = sorted(range(len(candidate_ids)), key=lambda i: scores[i], reverse=True)
    ranked = [candidate_ids[i] for i in order]
    return metrics_for_query(ranked, set(positive_ids), tuple(ks))


def first_stage_metrics(candidate_ids: Sequence[str], positive_ids,
                        ks: Sequence[int] = (10,)) -> Dict[str, float]:
    """Metrics for the candidate list *as given* (the first-stage order)."""
    return metrics_for_query(list(candidate_ids), set(positive_ids), tuple(ks))


def positive_in_top_k(candidate_ids: Sequence[str], scores: Sequence[float],
                      positive_ids, k: int) -> float:
    order = sorted(range(len(candidate_ids)), key=lambda i: scores[i], reverse=True)
    top = {candidate_ids[i] for i in order[:k]}
    return 1.0 if top & set(positive_ids) else 0.0


def oracle_metrics(candidate_ids: Sequence[str], positive_ids,
                   ks: Sequence[int] = (10,)) -> Dict[str, float]:
    """Best achievable ranking from this candidate set (all positives first)."""
    pos = [c for c in candidate_ids if c in set(positive_ids)]
    neg = [c for c in candidate_ids if c not in set(positive_ids)]
    return metrics_for_query(pos + neg, set(positive_ids), tuple(ks))


def aggregate_rows(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    return aggregate(list(rows))


# --------------------------------------------------------------- ML layer (lazy import)
def _load_cross_encoder(base_model: str, device: Optional[str], bf16: bool,
                        gradient_checkpointing: bool, use_lora: bool, num_labels: int = 1):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if bf16 else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=num_labels, torch_dtype=dtype, trust_remote_code=True)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if use_lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32))
    if device:
        model = model.to(device)
    return model, tok


def _encode_pairs(tok, pairs, template, max_len, device):
    texts = [template.format(query=q, document=d) for q, d in pairs]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return {k: v.to(device) for k, v in enc.items()} if device else enc


def train_pointwise_reranker(cfg, examples, output_dir, *, epochs=1, batch_size=16,
                             max_length=256, lr=2e-5, bf16=True, gradient_checkpointing=True,
                             use_lora=False, regression=False, device=None) -> Dict[str, Any]:
    """BCEWithLogits (binary) or MSE (regress teacher score) over (query, doc) pairs."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = _load_cross_encoder(cfg.model_name_or_path, device, bf16,
                                     gradient_checkpointing, use_lora)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss() if regression else torch.nn.BCEWithLogitsLoss()
    model.train()
    losses = []
    for _ in range(epochs):
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            enc = _encode_pairs(tok, [(e["query"], e["document"]) for e in batch],
                                cfg.input_template, max_length, device)
            labels = torch.tensor([e["label"] for e in batch], dtype=torch.float32, device=device)
            logits = model(**enc).logits.squeeze(-1)
            loss = loss_fn(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))
    model.save_pretrained(output_dir); tok.save_pretrained(output_dir)
    return {"status": "ok", "objective": "pointwise", "output_dir": output_dir,
            "num_examples": len(examples), "final_loss": losses[-1] if losses else None}


def train_pairwise_reranker(cfg, pairs, output_dir, *, epochs=1, batch_size=16, max_length=256,
                            lr=2e-5, margin=0.2, bf16=True, gradient_checkpointing=True,
                            use_lora=False, device=None) -> Dict[str, Any]:
    """MarginRankingLoss enforcing score(positive) > score(negative) + margin."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = _load_cross_encoder(cfg.model_name_or_path, device, bf16,
                                     gradient_checkpointing, use_lora)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MarginRankingLoss(margin=margin)
    model.train()
    losses = []
    for _ in range(epochs):
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            pos = _encode_pairs(tok, [(e["query"], e["positive"]) for e in batch],
                                cfg.input_template, max_length, device)
            neg = _encode_pairs(tok, [(e["query"], e["negative"]) for e in batch],
                                cfg.input_template, max_length, device)
            sp = model(**pos).logits.squeeze(-1)
            sn = model(**neg).logits.squeeze(-1)
            target = torch.ones_like(sp)
            loss = loss_fn(sp, sn, target)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))
    model.save_pretrained(output_dir); tok.save_pretrained(output_dir)
    return {"status": "ok", "objective": "pairwise", "output_dir": output_dir,
            "num_pairs": len(pairs), "final_loss": losses[-1] if losses else None}


def train_listwise_distilled_reranker(cfg, batches, output_dir, *, epochs=1, max_length=256,
                                      lr=2e-5, temperature=1.0, bf16=True,
                                      gradient_checkpointing=True, use_lora=False,
                                      device=None) -> Dict[str, Any]:
    """KLDiv of the student's candidate log-softmax toward the teacher's softmax target."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = _load_cross_encoder(cfg.model_name_or_path, device, bf16,
                                     gradient_checkpointing, use_lora)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    kl = torch.nn.KLDivLoss(reduction="batchmean")
    model.train()
    losses = []
    for _ in range(epochs):
        for b in batches:
            enc = _encode_pairs(tok, [(b["query"], d) for d in b["documents"]],
                                cfg.input_template, max_length, device)
            logits = model(**enc).logits.squeeze(-1)
            student_logp = torch.log_softmax(logits / max(temperature, 1e-6), dim=0)
            target = torch.tensor(b["target"], dtype=torch.float32, device=device)
            loss = kl(student_logp.unsqueeze(0), target.unsqueeze(0))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))
    model.save_pretrained(output_dir); tok.save_pretrained(output_dir)
    return {"status": "ok", "objective": "listwise", "output_dir": output_dir,
            "num_queries": len(batches), "final_loss": losses[-1] if losses else None}


def score_with_student_reranker(model_path, pairs, template, *, max_length=256,
                                batch_size=64, device=None) -> List[float]:
    """Score (query, document) pairs with a trained student cross-encoder. ML-only."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
    scores: List[float] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            enc = _encode_pairs(tok, pairs[start:start + batch_size], template, max_length, device)
            logits = model(**enc).logits.squeeze(-1)
            scores.extend([float(x) for x in logits.flatten().tolist()])
    return scores
