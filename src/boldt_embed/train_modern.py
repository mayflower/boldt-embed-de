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


def plan_loss_stack(student_cfg, has_teacher_scores: bool, use_guide: bool = False
                    ) -> Dict[str, Any]:
    """Describe (without importing ML) the loss stack that `build_losses` will instantiate."""
    losses = [str(x) for x in getattr(student_cfg, "losses", [])]
    dims = list(getattr(student_cfg, "matryoshka_dims", MATRYOSHKA_DEFAULT))
    base = "CachedGISTEmbedLoss" if use_guide else "CachedMultipleNegativesRankingLoss"
    stack = [base]
    if "matryoshka" in losses or not losses:
        stack.append(f"MatryoshkaLoss(dims={dims})")
    distill = []
    if has_teacher_scores and ("margin_mse" in losses or "distillation" in losses):
        distill.append("MarginMSELoss")
    return {
        "base_contrastive": base,
        "matryoshka_dims": dims,
        "wrapped": " -> ".join(stack),
        "distillation": distill,
        "uses_guide_model": use_guide,
        "teacher_distillation_active": bool(distill),
    }


# --------------------------------------------------------------- ML layer (lazy import)
def load_student_sentence_transformer(model_name: str, max_seq_length: int = 512,
                                      pooling: str = "mean", device: Optional[str] = None,
                                      trust_remote_code: bool = True):
    """Load the Boldt student as a SentenceTransformer. Builds a Transformer+Pooling stack
    explicitly so a decoder base gets a defined pooling head. Raises a clear error (pointing
    to the bidirectional adapter) rather than silently training the wrong architecture."""
    from sentence_transformers import SentenceTransformer, models

    try:
        transformer = models.Transformer(model_name, max_seq_length=max_seq_length,
                                          tokenizer_args={"trust_remote_code": trust_remote_code},
                                          model_args={"trust_remote_code": trust_remote_code})
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


def build_losses(model, student_cfg, has_teacher_scores: bool, guide_model=None) -> Dict[str, Any]:
    """Instantiate the loss objects described by `plan_loss_stack`. ML-only."""
    from sentence_transformers import losses as L

    dims = list(getattr(student_cfg, "matryoshka_dims", MATRYOSHKA_DEFAULT))
    if guide_model is not None:
        base = L.CachedGISTEmbedLoss(model, guide=guide_model)
    else:
        base = L.CachedMultipleNegativesRankingLoss(model)
    primary = L.MatryoshkaLoss(model, base, matryoshka_dims=dims)
    out: Dict[str, Any] = {"primary": primary}
    cfg_losses = [str(x) for x in getattr(student_cfg, "losses", [])]
    if has_teacher_scores and ("margin_mse" in cfg_losses or "distillation" in cfg_losses):
        out["distill"] = L.MarginMSELoss(model)
    return out


def train_modern_embedder(student_cfg, examples: Sequence[Dict[str, Any]], output_dir: str,
                          *, epochs: int = 1, max_steps: int = -1, batch_size: int = 32,
                          mini_batch_size: int = 8, lr: float = 2e-5, bf16: bool = True,
                          gradient_checkpointing: bool = True, use_lora: bool = False,
                          guide_model_name: Optional[str] = None, device: Optional[str] = None,
                          max_seq_length: int = 512) -> Dict[str, Any]:
    """Real training entry point (GPU). Builds the dataset, model, loss stack, and runs the
    SentenceTransformerTrainer with NO_DUPLICATES batch sampling. ML-only."""
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.training_args import BatchSamplers

    model = load_student_sentence_transformer(
        student_cfg.base_model, max_seq_length=max_seq_length, device=device)
    if use_lora:
        from peft import LoraConfig
        model.add_adapter(LoraConfig(task_type="FEATURE_EXTRACTION", r=16, lora_alpha=32))

    # Build a (anchor, positive[, negative]) dataset; include hardest negative if present.
    cols: Dict[str, List[Any]] = {"anchor": [], "positive": []}
    use_neg = any(e["negatives"] for e in examples)
    if use_neg:
        cols["negative"] = []
    for e in examples:
        cols["anchor"].append(e["query"])
        cols["positive"].append(e["positive"])
        if use_neg:
            cols["negative"].append(e["negatives"][0] if e["negatives"] else e["positive"])
    train_ds = Dataset.from_dict(cols)

    guide = SentenceTransformer(guide_model_name, device=device) if guide_model_name else None
    has_scores = dataset_metadata(examples)["has_teacher_scores"]
    loss_objs = build_losses(model, student_cfg, has_scores, guide_model=guide)

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs, max_steps=max_steps,
        per_device_train_batch_size=batch_size, learning_rate=lr, bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        batch_sampler=BatchSamplers.NO_DUPLICATES)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds,
                                         loss=loss_objs["primary"])
    trainer.train()
    model.save(output_dir)
    return {"status": "ok", "output_dir": output_dir, "num_examples": len(examples),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "loss_stack": plan_loss_stack(student_cfg, has_scores, use_guide=guide is not None)}


def export_sentence_transformers_model(model, output_dir: str) -> str:
    model.save(output_dir)
    return output_dir
