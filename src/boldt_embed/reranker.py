"""German cross-encoder reranker + hard-negative mining + distillation helpers.

The reranker encodes (query, document) together and outputs a relevance score (or a
`Ja`/`Nein` logit). It is used for production reranking, hard-negative mining, and as a
teacher for distillation into the bi-encoders. ``torch`` is imported lazily; mining and
distillation helpers are pure stdlib and accept a ``scorer`` callable so they are testable.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .config import RerankerConfig, load_reranker_config
from .textutil import normalize

Scorer = Callable[[str, str], float]


class Reranker:
    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None

    @classmethod
    def from_config(cls, path: str) -> "Reranker":
        return cls(load_reranker_config(path))

    # --- stdlib, no weights -------------------------------------------------
    def build_input(self, query: str, document: str) -> str:
        return self.config.input_template.replace("{query}", query).replace("{document}", document)

    def dry_run(self, query: str, documents: Sequence[str]) -> Dict[str, object]:
        inputs = [self.build_input(query, d) for d in documents]
        return {
            "status": "pass",
            "variant": self.config.variant,
            "base_model": self.config.model_name_or_path,
            "output_mode": self.config.output_mode,
            "labels": [self.config.positive_label, self.config.negative_label],
            "max_length": self.config.max_length,
            "hard_negative_sources": self.config.hard_negative_sources,
            "num_pairs": len(inputs),
            "sample_input": inputs[0] if inputs else None,
            "note": "Dry-run: cross-encoder input wiring validated. No weights loaded.",
        }

    # --- real path (needs torch/transformers) -------------------------------
    def _load(self):  # pragma: no cover - requires extras + weights
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install training extras: pip install -e '.[train]'") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name_or_path
        )
        self._model.eval()

    def score_pairs(self, query: str, documents: Sequence[str]) -> List[float]:  # pragma: no cover
        import torch

        self._load()
        inputs = [self.build_input(query, d) for d in documents]
        batch = self._tokenizer(inputs, padding=True, truncation=True,
                                max_length=self.config.max_length, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**batch).logits
        scores = logits[:, -1] if logits.shape[-1] > 1 else logits.squeeze(-1)
        return torch.sigmoid(scores).tolist()

    def rerank(self, query: str, documents: Sequence[str]) -> List[Tuple[int, float]]:  # pragma: no cover
        scores = self.score_pairs(query, documents)
        return sorted(enumerate(scores), key=lambda kv: kv[1], reverse=True)


# --------------------------------------------------------------- hard-negative mining
def mine_hard_negatives(
    query: str,
    candidates: Sequence[str],
    scorer: Scorer,
    *,
    positives: Sequence[str] = (),
    k: int = 3,
) -> List[str]:
    """Return the top-k highest-scoring candidates that are NOT positives.

    These are the documents the scorer ranks as most relevant but which are known to be
    wrong answers — the hardest negatives for contrastive training.
    """
    positive_norms = {normalize(p) for p in positives}
    scored = [
        (cand, scorer(query, cand))
        for cand in candidates
        if normalize(cand) not in positive_norms
    ]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return [cand for cand, _ in scored[:k]]


# ------------------------------------------------------------------------- distillation
def softmax(scores: Sequence[float], temperature: float = 1.0) -> List[float]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = [s / temperature for s in scores]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def distillation_soft_labels(teacher_scores: Sequence[float], temperature: float = 1.0) -> List[float]:
    """Teacher reranker scores -> soft target distribution for KL distillation."""
    return softmax(teacher_scores, temperature)


def margin_mse_target(positive_score: float, negative_score: float) -> float:
    """Teacher margin used by Margin-MSE distillation (student matches this margin)."""
    return positive_score - negative_score
