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


def train_modern_embedder(student_cfg, examples: Sequence[Dict[str, Any]], output_dir: str,
                          *, epochs: int = 1, max_steps: int = -1, batch_size: int = 32,
                          mini_batch_size: int = 8, lr: float = 2e-5, bf16: bool = True,
                          gradient_checkpointing: bool = True, use_lora: bool = False,
                          guide_model_name: Optional[str] = None, device: Optional[str] = None,
                          max_seq_length: int = 512, bidirectional: Optional[bool] = None,
                          use_distillation: Optional[bool] = None,
                          base_model: Optional[str] = None,
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

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs, max_steps=max_steps,
        per_device_train_batch_size=batch_size, learning_rate=lr, bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        batch_sampler=BatchSamplers.NO_DUPLICATES)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds,
                                         loss=loss_objs["primary"])
    trainer.train()
    model.save(output_dir)
    report = {"status": "ok", "output_dir": output_dir, "base_model": base,
              "num_examples": len(examples), "uses_explicit_negatives": use_neg,
              "bidirectional": bidirectional,
              "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "loss_stack": plan_loss_stack(student_cfg, has_scores, use_guide=guide is not None,
                                            use_distillation=use_distillation)}
    if extra_report:
        report.update(extra_report)
    return report


def export_sentence_transformers_model(model, output_dir: str) -> str:
    model.save(output_dir)
    return output_dir
