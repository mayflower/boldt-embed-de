"""Modern (2026) SentenceTransformers training path for the Boldt student (lazy ML imports).

Adds a distillation-ready embedding trainer on top of the teacher cache, using the modern
SBERT loss stack: a cached contrastive base (CachedMultipleNegativesRankingLoss, or
CachedGISTEmbedLoss with a guide model) wrapped in MatryoshkaLoss, plus optional MarginMSE
teacher-score distillation.

Layered like the rest of the workflow:

* **stdlib layer** — `build_train_dataset_from_teacher_cache`, `dataset_metadata`,
  `plan_loss_stack`. Used by `--dry-run`; imports no ML.
* **ML layer** — `load_student_sentence_transformer`, `build_losses`,
  `train_modern_embedder`, `export_sentence_transformers_model`. Lazy imports inside.

The legacy `train.py` (plain InfoNCE/MNRL/BCE) is kept as a baseline; this is additive.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

MATRYOSHKA_DEFAULT = [1024, 768, 512, 256, 128, 64]


# ---------------------------------------------------------------- stdlib: dataset/plan
def _best_positive(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    pos = [r for r in rows if r.get("positive") is True]
    if not pos:
        # fall back to label>0 if 'positive' wasn't set
        pos = [r for r in rows if isinstance(r.get("label"), (int, float)) and r["label"] > 0]
    if not pos:
        return None

    def score(r):
        return r.get("reranker_score") if r.get("reranker_score") is not None else (
            r.get("embedding_score") if r.get("embedding_score") is not None else 0.0)
    return max(pos, key=score)


def build_train_dataset_from_teacher_cache(cache_rows: Sequence[Dict[str, Any]]
                                           ) -> List[Dict[str, Any]]:
    """Group teacher-cache rows by query into training examples.

    Each example: {query, positive, negatives[], pos_score, neg_scores[]}. Negatives are the
    query's non-positive docs, hardest first (highest teacher score). Queries without any
    positive are skipped. Pure stdlib — accepts pair rows or full lists."""
    by_query: Dict[str, List[Dict[str, Any]]] = {}
    qtext: Dict[str, str] = {}
    for r in cache_rows:
        qid = str(r["query_id"])
        by_query.setdefault(qid, []).append(r)
        qtext[qid] = r.get("query", qtext.get(qid, ""))

    examples: List[Dict[str, Any]] = []
    for qid, rows in by_query.items():
        pos = _best_positive(rows)
        if pos is None:
            continue
        negs = [r for r in rows if r is not pos and r.get("positive") is not True]

        def nscore(r):
            return r.get("reranker_score") if r.get("reranker_score") is not None else (
                r.get("embedding_score") if r.get("embedding_score") is not None else float("-inf"))
        negs = sorted(negs, key=nscore, reverse=True)
        examples.append({
            "query": qtext[qid],
            "positive": pos["document"],
            "negatives": [n["document"] for n in negs],
            "pos_score": pos.get("reranker_score") if pos.get("reranker_score") is not None
            else pos.get("embedding_score"),
            "neg_scores": [(n.get("reranker_score") if n.get("reranker_score") is not None
                            else n.get("embedding_score")) for n in negs],
        })
    return examples


def build_train_dataset_from_hardneg(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build training examples from a mined hard-negative file (schema from
    negative_mining_2026: {query, positive, negatives:[{document, reranker_teacher_score}]}).
    Negatives are kept in mined (hardest-first) order. Pure stdlib."""
    examples: List[Dict[str, Any]] = []
    for r in rows:
        negs = r.get("negatives") or []
        examples.append({
            "query": r.get("query", ""),
            "positive": r.get("positive", ""),
            "negatives": [n.get("document", "") for n in negs],
            "pos_score": None,
            "neg_scores": [n.get("reranker_teacher_score", n.get("embedding_teacher_score"))
                           for n in negs],
        })
    return [e for e in examples if e["query"] and e["positive"]]


def dataset_metadata(examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n_neg = sum(len(e["negatives"]) for e in examples)
    with_neg = sum(1 for e in examples if e["negatives"])
    with_scores = sum(1 for e in examples if e.get("pos_score") is not None)
    return {
        "num_examples": len(examples),
        "num_with_negatives": with_neg,
        "total_negatives": n_neg,
        "avg_negatives": round(n_neg / len(examples), 3) if examples else 0.0,
        "num_with_teacher_scores": with_scores,
        "has_teacher_scores": with_scores > 0,
    }


def plan_loss_stack(student_cfg, has_teacher_scores: bool, use_guide: bool = False,
                    use_distillation: Optional[bool] = None) -> Dict[str, Any]:
    """Describe (without importing ML) the loss stack that `build_losses` will instantiate.
    ``use_distillation``: None = auto (teacher scores present AND config requests it); True/False
    forces it on/off."""
    losses = [str(x) for x in getattr(student_cfg, "losses", [])]
    dims = list(getattr(student_cfg, "matryoshka_dims", MATRYOSHKA_DEFAULT))
    base = "CachedGISTEmbedLoss" if use_guide else "CachedMultipleNegativesRankingLoss"
    stack = [base]
    if "matryoshka" in losses or not losses:
        stack.append(f"MatryoshkaLoss(dims={dims})")
    auto_distill = has_teacher_scores and ("margin_mse" in losses or "distillation" in losses)
    distill_on = auto_distill if use_distillation is None else (use_distillation and has_teacher_scores)
    distill = ["MarginMSELoss"] if distill_on else []
    return {
        "base_contrastive": base,
        "matryoshka_dims": dims,
        "wrapped": " -> ".join(stack),
        "distillation": distill,
        "uses_guide_model": use_guide,
        "teacher_distillation_active": bool(distill),
    }


def plan_edge_spectrum_regularizer(reg_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Stdlib plan for the OPTIONAL, default-OFF v7 edge-spectrum regularizer. Reports whether it
    would be active. EXPERIMENTAL — this is NOT the paper's evaluated method; it only discourages
    pooled embeddings from carrying energy in the dropped edge singular subspace."""
    reg = reg_cfg or {}
    enabled = bool(reg.get("enabled", False))
    lam = float(reg.get("lambda", 0.0) or 0.0)
    active = enabled and lam > 0.0
    return {
        "name": "edge_spectrum_regularizer",
        "enabled": enabled,
        "lambda": lam,
        "active": active,
        "embed_filter_artifact": reg.get("embed_filter_artifact"),
        "apply_to": reg.get("apply_to", "pooled_embeddings"),
        "normalize_before": bool(reg.get("normalize_before", False)),
        "status": "active" if active else ("enabled_zero_lambda" if enabled else "disabled"),
        "note": "experimental; not the paper's evaluated method; default off",
    }


# ---------------------------------------------------------------- v5 dense RAG (stdlib)
def _margin_bucket(m: float) -> str:
    if m < 0:
        return "<0"
    if m >= 5:
        return ">=5"
    return f"{int(m)}-{int(m) + 1}"


def build_v5_dense_dataset(pairs: Sequence[Dict[str, Any]],
                           hardnegs: Optional[Sequence[Dict[str, Any]]] = None,
                           teacher_scores: Optional[Sequence[Dict[str, Any]]] = None,
                           *, teacher_threshold: float = 4.0) -> Dict[str, Any]:
    """Build the v5 dense-RAG training set from rag_pairs (+ optional webfaq2 hard-neg triplets +
    optional embedding-teacher scores). Pure stdlib. HARD-FAILS (via ``errors``) on public-benchmark
    / eval leakage. Synthetic pairs flagged ``must_teacher_validate`` are kept only when a
    ``teacher_score`` clears ``teacher_threshold`` (provisional rows excluded, not silently trained).
    """
    from .v5_data_mixer import leakage_reason
    hardnegs = list(hardnegs or [])
    teacher_scores = list(teacher_scores or [])
    errors: List[str] = []
    for i, r in enumerate(list(pairs) + hardnegs):
        if isinstance(r, dict):
            reason = leakage_reason(r)
            if reason:
                errors.append(f"row {i} ({r.get('source_id') or r.get('query', '?')}): "
                              f"public-benchmark/eval leakage ({reason})")

    tmap: Dict[tuple, float] = {}
    has_distill_vectors = False
    for s in teacher_scores:
        key = (str(s.get("query") or s.get("query_id")), str(s.get("document") or s.get("doc_id")))
        if s.get("teacher_score") is not None:
            tmap[key] = float(s["teacher_score"])
        if s.get("teacher_vector") or s.get("embedding"):
            has_distill_vectors = True

    def _pos(r):
        return r.get("positive") or r.get("document") or r.get("answer")

    validated = provisional_excluded = real_pairs = 0
    examples: List[Dict[str, Any]] = []
    domain_mix: Dict[str, int] = {}
    for r in pairs:
        q, p = r.get("query"), _pos(r)
        if not (isinstance(q, str) and q.strip() and isinstance(p, str) and p.strip()):
            continue
        if r.get("must_teacher_validate") is True:
            ts = r.get("teacher_score")
            if not (isinstance(ts, (int, float)) and not isinstance(ts, bool) and ts >= teacher_threshold):
                provisional_excluded += 1
                continue
            validated += 1
        else:
            real_pairs += 1
        dom = r.get("domain", "unknown")
        domain_mix[dom] = domain_mix.get(dom, 0) + 1
        examples.append({"query": q, "positive": p, "negatives": [],
                         "pos_score": tmap.get((str(q), str(p))), "neg_scores": [], "domain": dom})

    grouped: Dict[tuple, Dict[str, Any]] = {}
    for r in hardnegs:
        q, p, n = r.get("query"), r.get("positive"), r.get("negative")
        if not (q and p and n):
            continue
        g = grouped.setdefault((q, p), {"query": q, "positive": p, "negatives": [], "neg_scores": [],
                                        "margins": [], "domain": r.get("domain", "webfaq2"),
                                        "pos_score": r.get("positive_score")})
        g["negatives"].append(n)
        g["neg_scores"].append(r.get("negative_score"))
        g["margins"].append(r.get("teacher_margin"))

    margins_all: List[float] = []
    margin_hist: Dict[str, int] = {}
    for g in grouped.values():
        ms = [m for m in g["margins"] if isinstance(m, (int, float)) and not isinstance(m, bool)]
        margins_all += ms
        for m in ms:
            margin_hist[_margin_bucket(m)] = margin_hist.get(_margin_bucket(m), 0) + 1
        domain_mix[g["domain"]] = domain_mix.get(g["domain"], 0) + 1
        examples.append({"query": g["query"], "positive": g["positive"], "negatives": g["negatives"],
                         "pos_score": g["pos_score"], "neg_scores": g["neg_scores"], "domain": g["domain"]})

    has_teacher_scores = bool(tmap) or bool(margins_all) or any(
        e.get("pos_score") is not None for e in examples)
    report = {
        "num_pairs_in": len(list(pairs)), "num_hardneg_triplets_in": len(hardnegs),
        "examples": len(examples),
        "domain_mix": dict(sorted(domain_mix.items())),
        "teacher_validation": {"validated_synthetic": validated, "real_pairs": real_pairs,
                               "provisional_excluded": provisional_excluded,
                               "teacher_threshold": teacher_threshold},
        "hard_negatives": {"triplets": len(hardnegs), "queries_with_hardnegs": len(grouped),
                           "avg_margin": round(sum(margins_all) / len(margins_all), 4) if margins_all else None,
                           "margin_distribution": dict(sorted(margin_hist.items()))},
        "has_teacher_scores": has_teacher_scores,
        "has_distill_vectors": has_distill_vectors,
        "leakage_rows": len(errors),
    }
    return {"examples": examples, "report": report, "errors": errors}


def plan_v5_dense_loss_stack(*, has_teacher_scores: bool, has_distill_vectors: bool,
                             matryoshka_dims: Optional[Sequence[int]] = None,
                             use_guide: bool = False) -> Dict[str, Any]:
    """v5 dense loss stack (stdlib, no ML): CachedMNRL -> Matryoshka, + MarginMSE (teacher scores)
    + optional EmbedDistill (Qwen3-Embedding-8B vectors), NO_DUPLICATES sampler."""
    dims = list(matryoshka_dims or MATRYOSHKA_DEFAULT)
    base = "CachedGISTEmbedLoss" if use_guide else "CachedMultipleNegativesRankingLoss"
    wrapped = [base, f"MatryoshkaLoss(dims={dims})"]
    margin = ["MarginMSELoss"] if has_teacher_scores else []
    distill = ["EmbedDistillLoss(MSELoss to Qwen3-Embedding-8B vectors)"] if has_distill_vectors else []
    return {
        "base_contrastive": base,
        "wrapped": " -> ".join(wrapped),
        "matryoshka_dims": dims,
        "margin_mse": margin,
        "embed_distill": distill,
        "batch_sampler": "NO_DUPLICATES",
        "loss_stack": wrapped + margin + distill,
    }


def v5_dense_run_card(dataset_report: Dict[str, Any], loss_plan: Dict[str, Any], *, run_id: str,
                      model: str, output: str, max_steps: int, bf16: bool,
                      gradient_checkpointing: bool, lora: bool = False,
                      timestamp: Optional[str] = None) -> Dict[str, Any]:
    return {
        "run_id": run_id, "model": model, "lora": lora, "output": output,
        "max_steps": max_steps, "bf16": bf16, "gradient_checkpointing": gradient_checkpointing,
        "loss_stack": loss_plan["loss_stack"], "batch_sampler": loss_plan["batch_sampler"],
        "matryoshka_dims": loss_plan["matryoshka_dims"],
        "num_examples": dataset_report["examples"], "domain_mix": dataset_report["domain_mix"],
        "hard_negative_margins": dataset_report["hard_negatives"],
        "teacher_validation": dataset_report["teacher_validation"],
        "has_teacher_scores": dataset_report["has_teacher_scores"],
        "has_distill_vectors": dataset_report["has_distill_vectors"],
        "timestamp": timestamp,
        "purpose": "v5 dense RAG retriever: improve first-stage recall + candidate lists for reranking",
    }


# ---------------------------------------------------------------- v6 dense RAG (stdlib)
# v6 target is FIRST-STAGE RECALL (Recall@50/100 + candidate-list coverage), NOT reranker lift.
FAQ_DOMAINS = ("faq_real", "faq", "webfaq", "webfaq2", "faq_synthetic")


def _is_faq_domain(domain: Any) -> bool:
    d = str(domain or "").lower()
    return d in FAQ_DOMAINS or d.startswith("faq") or "faq" in d


def domain_balanced_examples(examples: Sequence[Dict[str, Any]], *, faq_cap: float = 0.5,
                             faq_domains: Sequence[str] = FAQ_DOMAINS):
    """Cap the FAQ share to <= ``faq_cap`` of the kept set, then round-robin interleave domains so
    NO_DUPLICATES batches are domain-balanced (FAQ never dominates a batch). Deterministic: FAQ is
    stride-sampled (diversity-preserving) and domains are cycled in sorted order. Returns
    (ordered_examples, report). Pure stdlib."""
    faqset = {d.lower() for d in faq_domains}

    def isfaq(e):
        d = str(e.get("domain") or "").lower()
        return d in faqset or d.startswith("faq") or "faq" in d

    faq = [e for e in examples if isfaq(e)]
    nonfaq = [e for e in examples if not isfaq(e)]
    before: Dict[str, int] = {}
    for e in examples:
        before[str(e.get("domain") or "unknown")] = before.get(str(e.get("domain") or "unknown"), 0) + 1

    kept_faq, capped = faq, False
    if faq and 0.0 < faq_cap < 1.0:
        max_faq = int((faq_cap / (1.0 - faq_cap)) * len(nonfaq))
        if len(faq) > max_faq:
            stride = len(faq) / max_faq if max_faq > 0 else 0.0
            kept_faq = [faq[int(i * stride)] for i in range(max_faq)] if max_faq > 0 else []
            capped = True
    kept = kept_faq + nonfaq

    kb: Dict[str, List[Dict[str, Any]]] = {}
    for e in kept:
        kb.setdefault(str(e.get("domain") or "unknown"), []).append(e)
    domains = sorted(kb)
    idx = {d: 0 for d in domains}
    order: List[Dict[str, Any]] = []
    remaining = len(kept)
    while remaining > 0:
        for d in domains:
            if idx[d] < len(kb[d]):
                order.append(kb[d][idx[d]])
                idx[d] += 1
                remaining -= 1
    after = {d: len(v) for d, v in sorted(kb.items())}
    return order, {
        "faq_cap": faq_cap, "faq_capped": capped,
        "faq_share_before": round(len(faq) / len(examples), 4) if examples else 0.0,
        "faq_share_after": round(len(kept_faq) / len(kept), 4) if kept else 0.0,
        "examples_before": len(examples), "examples_after": len(kept),
        "by_domain_before": dict(sorted(before.items())), "by_domain_after": after,
        "batch_strategy": "domain round-robin (balanced) + NO_DUPLICATES",
    }


def build_v6_dense_dataset(pairs: Sequence[Dict[str, Any]],
                           hardnegs: Optional[Sequence[Dict[str, Any]]] = None,
                           teacher_scores: Optional[Sequence[Dict[str, Any]]] = None,
                           *, faq_cap: float = 0.5, teacher_threshold: float = 4.0) -> Dict[str, Any]:
    """v6 dense-RAG dataset: same leakage-fail-closed + teacher-validation + hard-neg-margin handling
    as v5, then FAQ-capped, domain-balanced ordering for first-stage-recall training. Pure stdlib."""
    base = build_v5_dense_dataset(pairs, hardnegs, teacher_scores, teacher_threshold=teacher_threshold)
    ordered, balance = domain_balanced_examples(base["examples"], faq_cap=faq_cap)
    report = dict(base["report"])
    report["domain_balance"] = balance
    report["examples"] = len(ordered)
    report["domain_mix"] = balance["by_domain_after"]
    return {"examples": ordered, "report": report, "errors": base["errors"]}


def v6_dense_run_card(dataset_report: Dict[str, Any], loss_plan: Dict[str, Any], *, run_id: str,
                      model: str, output: str, max_steps: int, batch_size: int, bf16: bool,
                      gradient_checkpointing: bool, lora: bool = False,
                      timestamp: Optional[str] = None) -> Dict[str, Any]:
    card = v5_dense_run_card(dataset_report, loss_plan, run_id=run_id, model=model, output=output,
                             max_steps=max_steps, bf16=bf16,
                             gradient_checkpointing=gradient_checkpointing, lora=lora,
                             timestamp=timestamp)
    card["batch_size"] = batch_size
    card["domain_balance"] = dataset_report.get("domain_balance")
    card["target_metric"] = "Recall@50/100 + candidate-list coverage (NOT reranker lift)"
    card["purpose"] = ("v6 dense German RAG retriever: raise first-stage recall so candidate lists "
                       "contain the positives a reranker would otherwise never see")
    return card


# ---------------------------------------------------------------- v6.1 dense top-50 (stdlib)
def build_v6_1_dense_dataset(pairs: Sequence[Dict[str, Any]],
                             hardnegs: Sequence[Dict[str, Any]], *,
                             max_triplets_per_query: Optional[int] = None,
                             top50: int = 50, window: int = 200) -> Dict[str, Any]:
    """v6.1 dense Recall@50 dataset (pure stdlib). Returns contrastive PAIRS (query, positive) from
    rag_pairs + RANK-PROMOTION TRIPLETS (query, positive, top50-blocker) from the dense top-50 hard
    negatives — only for queries whose positive sits at dense rank ``top50``..``window``. Fails
    closed on public-benchmark/eval leakage. The triplets' negatives are already teacher-vetted (the
    mining veto used teacher margins)."""
    from .v5_data_mixer import leakage_reason
    errors: List[str] = []
    pair_examples: List[Dict[str, Any]] = []
    for i, r in enumerate(pairs):
        reason = leakage_reason(r)
        if reason:
            errors.append(f"pair {i} ({r.get('source') or r.get('query', '?')}): leakage ({reason})")
            continue
        q = r.get("query")
        p = r.get("positive") or r.get("document")
        if isinstance(q, str) and q.strip() and isinstance(p, str) and p.strip():
            pair_examples.append({"query": q, "positive": p, "domain": r.get("domain", "unknown")})

    triplets: List[Dict[str, Any]] = []
    rank_51_100 = rank_101_200 = 0
    domain_mix: Dict[str, int] = {}
    margins: List[float] = []
    for i, r in enumerate(hardnegs):
        reason = leakage_reason(r)
        if reason:
            errors.append(f"hardneg {i} ({r.get('query_id', '?')}): leakage ({reason})")
            continue
        pr = r.get("positive_rank_v6")
        if not (isinstance(pr, int) and top50 < pr <= window):
            continue
        q, pos = r.get("query"), r.get("positive")
        if not (q and pos):
            continue
        dom = r.get("domain", "unknown")
        negs = r.get("negatives") or []
        if max_triplets_per_query is not None:
            negs = negs[:max_triplets_per_query]
        used = 0
        for n in negs:
            neg = n.get("text")
            if not neg:
                continue
            triplets.append({"query": q, "positive": pos, "negative": neg, "domain": dom,
                             "positive_rank_v6": pr, "negative_rank_v6": n.get("negative_rank_v6"),
                             "teacher_score": n.get("teacher_score"),
                             "margin_to_positive": n.get("margin_to_positive")})
            if n.get("margin_to_positive") is not None:
                margins.append(float(n["margin_to_positive"]))
            used += 1
        if used:
            domain_mix[dom] = domain_mix.get(dom, 0) + 1
            if pr <= 100:
                rank_51_100 += 1
            else:
                rank_101_200 += 1

    margin_hist: Dict[str, int] = {}
    for m in margins:
        margin_hist[_margin_bucket(m)] = margin_hist.get(_margin_bucket(m), 0) + 1
    report = {
        "pair_examples": len(pair_examples), "rank_promotion_triplets": len(triplets),
        "rank_promotion_queries": rank_51_100 + rank_101_200,
        "positive_rank_51_100": rank_51_100, "positive_rank_101_200": rank_101_200,
        "domain_mix": dict(sorted(domain_mix.items())),
        "has_teacher_margins": bool(margins),
        "hard_negative_margins": {"with_margin": len(margins),
                                  "avg": round(sum(margins) / len(margins), 4) if margins else None,
                                  "distribution": dict(sorted(margin_hist.items()))},
        "leakage_rows": len(errors), "window": window, "top50": top50,
    }
    return {"pair_examples": pair_examples, "triplet_examples": triplets, "report": report,
            "errors": errors}


def plan_v6_1_loss_stack(*, has_teacher_margins: bool,
                         matryoshka_dims: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """v6.1 loss stack (stdlib): CachedMNRL -> Matryoshka, with RANK-PROMOTION realized as CMNRL over
    (query, positive, top50-blocker) triplets (pushes sim(q,pos) above the blockers). MarginMSE is
    teacher-margin-prepared (the triplets are teacher-vetted) but not wired as a separate loss in this
    run. NO_DUPLICATES sampler."""
    dims = list(matryoshka_dims or MATRYOSHKA_DEFAULT)
    stack = ["CachedMultipleNegativesRankingLoss", f"MatryoshkaLoss(dims={dims})",
             "RankPromotion(CMNRL over query/positive/top50_blocker triplets)"]
    return {
        "base_contrastive": "CachedMultipleNegativesRankingLoss",
        "matryoshka_dims": dims,
        "rank_promotion": ("CMNRL with explicit top-50 false-positive negatives — the positive is "
                           "pushed above the docs that currently outrank it (Recall@50 lever)"),
        "margin_mse": (["MarginMSELoss(teacher margins; data-level teacher-vetted negatives)"]
                       if has_teacher_margins else []),
        "margin_mse_wired_as_separate_loss": False,
        "batch_sampler": "NO_DUPLICATES",
        "loss_stack": stack,
    }


def v6_1_dense_run_card(dataset_report: Dict[str, Any], loss_plan: Dict[str, Any], *, run_id: str,
                        base_model: str, output: str, max_steps: int, batch_size: int, bf16: bool,
                        gradient_checkpointing: bool, timestamp: Optional[str] = None) -> Dict[str, Any]:
    return {
        "run_id": run_id, "base_checkpoint": base_model, "output": output, "max_steps": max_steps,
        "batch_size": batch_size, "bf16": bf16, "gradient_checkpointing": gradient_checkpointing,
        "timestamp": timestamp, "loss_stack": loss_plan["loss_stack"],
        "rank_promotion": loss_plan["rank_promotion"], "matryoshka_dims": loss_plan["matryoshka_dims"],
        "batch_sampler": loss_plan["batch_sampler"],
        "margin_mse": loss_plan["margin_mse"],
        "margin_mse_wired_as_separate_loss": loss_plan["margin_mse_wired_as_separate_loss"],
        "pair_examples": dataset_report["pair_examples"],
        "rank_promotion_triplets": dataset_report["rank_promotion_triplets"],
        "rank_promotion_queries": dataset_report["rank_promotion_queries"],
        "positive_rank_51_100": dataset_report["positive_rank_51_100"],
        "positive_rank_101_200": dataset_report["positive_rank_101_200"],
        "domain_mix": dataset_report["domain_mix"],
        "hard_negative_margins": dataset_report["hard_negative_margins"],
        "reranker_trained": False,
        "target_metric": "WebFAQ Recall@50 >= 0.90 while preserving Recall@100/guardrails/Matryoshka",
        "purpose": ("v6.1 dense retriever: pull WebFAQ positives from ranks 51-200 into the top-50 "
                    "via rank-promotion over dense-v6 top-50 blockers; DENSE-ONLY, no reranker"),
    }


# --------------------------------------------------------------- ML layer (lazy import)
def apply_bidirectional_to_st_module(transformer_module) -> None:
    """Apply the LLM2Vec bidirectional mask patch to a SentenceTransformers Transformer
    module's wrapped HF model (``transformer_module.auto_model``). ML-only."""
    from .train import enable_bidirectional
    enable_bidirectional(transformer_module.auto_model)


def apply_bidirectional_to_st(st_model) -> None:
    """Re-apply the bidirectional patch to a loaded SentenceTransformer (its first module is
    the Transformer). Eval code calls this so a saved bidirectional student is actually
    bidirectional at inference — the patch is runtime, not persisted in weights."""
    apply_bidirectional_to_st_module(st_model[0])


def load_student_sentence_transformer(model_name: str, max_seq_length: int = 512,
                                      pooling: str = "mean", device: Optional[str] = None,
                                      trust_remote_code: bool = True, bidirectional: bool = False):
    """Load the Boldt student as a SentenceTransformer. Builds a Transformer+Pooling stack
    explicitly so a decoder base gets a defined pooling head. Raises a clear error (pointing
    to the bidirectional adapter) rather than silently training the wrong architecture.

    ``bidirectional=True`` loads with eager attention and applies the LLM2Vec mask patch to the
    wrapped decoder so every token attends to all non-pad positions. NOTE: the patch is a
    runtime modification (not saved weights) — eval code must re-apply it on load
    (``apply_bidirectional_to_st``)."""
    from sentence_transformers import SentenceTransformer, models

    model_args = {"trust_remote_code": trust_remote_code}
    if bidirectional:
        model_args["attn_implementation"] = "eager"  # custom 4D mask needs eager attention
    try:
        transformer = models.Transformer(model_name, max_seq_length=max_seq_length,
                                          tokenizer_args={"trust_remote_code": trust_remote_code},
                                          model_args=model_args)
        if bidirectional:
            apply_bidirectional_to_st_module(transformer)
        pool = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode=pooling)
        st = SentenceTransformer(modules=[transformer, pool], device=device)
        return st
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not build a SentenceTransformer from '{model_name}': {exc}. "
            "For the bidirectional student, prepare it first with "
            "boldt_embed.llm2vec_boldt (Prompt 5) and load the adapted checkpoint here. "
            "Not falling back silently — that would train the wrong architecture."
        ) from exc


def build_losses(model, student_cfg, has_teacher_scores: bool, guide_model=None,
                 use_distillation: Optional[bool] = None) -> Dict[str, Any]:
    """Instantiate the loss objects described by `plan_loss_stack`. ML-only."""
    from sentence_transformers import losses as L

    dims = list(getattr(student_cfg, "matryoshka_dims", MATRYOSHKA_DEFAULT))
    if guide_model is not None:
        base = L.CachedGISTEmbedLoss(model, guide=guide_model)
    else:
        base = L.CachedMultipleNegativesRankingLoss(model)
    primary = L.MatryoshkaLoss(model, base, matryoshka_dims=dims)
    out: Dict[str, Any] = {"primary": primary}
    plan = plan_loss_stack(student_cfg, has_teacher_scores, use_guide=guide_model is not None,
                           use_distillation=use_distillation)
    if plan["teacher_distillation_active"]:
        out["distill"] = L.MarginMSELoss(model)
    return out


def edge_spectrum_penalty(pooled, bulk_basis, normalize_before: bool = False):
    """Mean squared norm of the part of ``pooled`` OUTSIDE the kept bulk subspace — the residual
    after reconstructing from the bulk basis. ``bulk_basis``: [H, K] (orthonormal columns). Lazy
    torch. Used only by the opt-in v7 edge-spectrum regularizer."""
    x = pooled
    if normalize_before:
        import torch.nn.functional as F
        x = F.normalize(x, dim=1)
    recon = (x @ bulk_basis) @ bulk_basis.t()      # projection back onto the bulk subspace
    residual = x - recon
    return residual.pow(2).sum(dim=1).mean()


def make_edge_regularized_loss(base_loss, model, bulk_basis, lam: float,
                               normalize_before: bool = False):
    """Wrap an SBERT loss so it adds ``lam * edge_spectrum_penalty(anchor_pooled)``. Lazy torch.
    EXPERIMENTAL — instantiated ONLY when the regularizer is explicitly enabled (default off)."""
    import torch.nn as nn

    class _EdgeRegLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base_loss
            self.model = model
            self.register_buffer("basis", bulk_basis)
            self.lam = float(lam)
            self.normalize_before = bool(normalize_before)

        def forward(self, sentence_features, labels=None):
            loss = self.base(sentence_features, labels)
            pooled = self.model(sentence_features[0])["sentence_embedding"]
            pen = edge_spectrum_penalty(pooled, self.basis.to(pooled.device, pooled.dtype),
                                        self.normalize_before)
            return loss + self.lam * pen

    return _EdgeRegLoss()


def train_modern_embedder(student_cfg, examples: Sequence[Dict[str, Any]], output_dir: str,
                          *, epochs: int = 1, max_steps: int = -1, batch_size: int = 32,
                          mini_batch_size: int = 8, lr: float = 2e-5, bf16: bool = True,
                          gradient_checkpointing: bool = True, use_lora: bool = False,
                          guide_model_name: Optional[str] = None, device: Optional[str] = None,
                          max_seq_length: int = 512, bidirectional: Optional[bool] = None,
                          use_distillation: Optional[bool] = None,
                          base_model: Optional[str] = None,
                          edge_reg: Optional[Dict[str, Any]] = None,
                          extra_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Real training entry point (GPU). Builds the dataset, model, loss stack, and runs the
    SentenceTransformerTrainer with NO_DUPLICATES batch sampling. ML-only.

    ``bidirectional`` overrides the cfg variant; ``base_model`` overrides cfg.base_model (e.g.
    an MNTP-adapted checkpoint); ``use_distillation`` forces MarginMSE on/off."""
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.training_args import BatchSamplers

    if bidirectional is None:
        bidirectional = getattr(student_cfg, "student_variant", "causal") == "bidirectional"
    base = base_model or student_cfg.base_model
    model = load_student_sentence_transformer(
        base, max_seq_length=max_seq_length, device=device, bidirectional=bidirectional)
    if use_lora:
        from peft import LoraConfig
        model.add_adapter(LoraConfig(task_type="FEATURE_EXTRACTION", r=16, lora_alpha=32))

    # (anchor, positive[, negative]): only emit a negative column when EVERY example has one
    # (mixing real negatives with positive-as-negative placeholders would corrupt the loss).
    cols: Dict[str, List[Any]] = {"anchor": [], "positive": []}
    use_neg = bool(examples) and all(e["negatives"] for e in examples)
    if use_neg:
        cols["negative"] = []
    for e in examples:
        cols["anchor"].append(e["query"])
        cols["positive"].append(e["positive"])
        if use_neg:
            cols["negative"].append(e["negatives"][0])
    train_ds = Dataset.from_dict(cols)

    guide = SentenceTransformer(guide_model_name, device=device) if guide_model_name else None
    has_scores = dataset_metadata(examples)["has_teacher_scores"]
    loss_objs = build_losses(model, student_cfg, has_scores, guide_model=guide,
                             use_distillation=use_distillation)
    primary_loss = loss_objs["primary"]
    reg_plan = plan_edge_spectrum_regularizer(edge_reg)
    if reg_plan["active"]:   # opt-in only — default config leaves this disabled
        from boldt_embed.embed_filter import load_embed_filter_basis
        hdim = model.get_sentence_embedding_dimension()
        basis, _ = load_embed_filter_basis(reg_plan["embed_filter_artifact"],
                                           expected_hidden_dim=hdim)
        primary_loss = make_edge_regularized_loss(
            primary_loss, model, basis.to(model.device), reg_plan["lambda"],
            reg_plan["normalize_before"])

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs, max_steps=max_steps,
        per_device_train_batch_size=batch_size, learning_rate=lr, bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        batch_sampler=BatchSamplers.NO_DUPLICATES)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds,
                                         loss=primary_loss)
    trainer.train()
    model.save(output_dir)
    report = {"status": "ok", "output_dir": output_dir, "base_model": base,
              "num_examples": len(examples), "uses_explicit_negatives": use_neg,
              "bidirectional": bidirectional,
              "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "edge_spectrum_regularizer": reg_plan,
              "loss_stack": plan_loss_stack(student_cfg, has_scores, use_guide=guide is not None,
                                            use_distillation=use_distillation)}
    if extra_report:
        report.update(extra_report)
    return report


def train_v6_1_dense_embedder(base_model: str, pair_examples: Sequence[Dict[str, Any]],
                              triplet_examples: Sequence[Dict[str, Any]], output_dir: str, *,
                              matryoshka_dims: Optional[Sequence[int]] = None, epochs: int = 1,
                              max_steps: int = -1, batch_size: int = 64, lr: float = 2e-5,
                              max_seq_length: int = 256, bf16: bool = True,
                              gradient_checkpointing: bool = False, device: Optional[str] = None
                              ) -> Dict[str, Any]:
    """v6.1 dense training (GPU). Continues from ``base_model`` and trains CachedMNRL -> Matryoshka
    over a DatasetDict of contrastive PAIRS (query, positive) + RANK-PROMOTION TRIPLETS
    (query, positive, top50-blocker). The triplets' explicit hard negatives ARE the rank-promotion
    loss: CMNRL pushes sim(q,positive) above the blockers. NO_DUPLICATES sampler. ML-only."""
    import torch
    from datasets import Dataset, DatasetDict
    from sentence_transformers import (SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers import losses as L
    from sentence_transformers.training_args import BatchSamplers

    dims = list(matryoshka_dims or MATRYOSHKA_DEFAULT)
    model = load_student_sentence_transformer(base_model, max_seq_length=max_seq_length, device=device)
    loss = L.MatryoshkaLoss(model, L.CachedMultipleNegativesRankingLoss(model), matryoshka_dims=dims)

    parts = {}
    if pair_examples:
        parts["pairs"] = Dataset.from_dict({"anchor": [e["query"] for e in pair_examples],
                                            "positive": [e["positive"] for e in pair_examples]})
    if triplet_examples:
        parts["triplets"] = Dataset.from_dict({
            "anchor": [e["query"] for e in triplet_examples],
            "positive": [e["positive"] for e in triplet_examples],
            "negative": [e["negative"] for e in triplet_examples]})
    if not parts:
        raise ValueError("no training examples (pairs or triplets) for v6.1")
    train_dataset = DatasetDict(parts) if len(parts) > 1 else next(iter(parts.values()))

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs, max_steps=max_steps,
        per_device_train_batch_size=batch_size, learning_rate=lr, bf16=bf16,
        gradient_checkpointing=gradient_checkpointing, batch_sampler=BatchSamplers.NO_DUPLICATES)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_dataset,
                                         loss=loss)
    trainer.train()
    model.save(output_dir)
    return {"status": "ok", "output_dir": output_dir, "base_model": base_model,
            "datasets": sorted(parts), "num_pairs": len(pair_examples),
            "num_rank_promotion_triplets": len(triplet_examples), "matryoshka_dims": dims,
            "max_steps": max_steps, "rank_promotion": "CMNRL explicit top-50 blocker negatives",
            "reranker_trained": False,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def export_sentence_transformers_model(model, output_dir: str) -> str:
    model.save(output_dir)
    return output_dir
