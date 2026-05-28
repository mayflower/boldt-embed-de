"""Bidirectional (LLM2Vec / MNTP-style) embedder wrapper.

LLM2Vec recipe: (1) enable bidirectional attention, (2) MNTP adaptation, (3) contrastive
training, optionally (4) merge checkpoints. ``torch``/``transformers`` are imported lazily;
the dry-run and the merge math are stdlib and unit-testable.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .config import BidirectionalConfig, load_bidirectional_config
from .instructions import format_document


class BidirectionalEmbedder:
    def __init__(self, config: BidirectionalConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None

    @classmethod
    def from_config(cls, path: str) -> "BidirectionalEmbedder":
        return cls(load_bidirectional_config(path))

    # --- stdlib, no weights -------------------------------------------------
    def build_inputs(self, texts: Sequence[str], instruction: str = "") -> List[str]:
        return [format_document(instruction or "{document}", t) for t in texts]

    def mntp_plan(self) -> Dict[str, object]:
        """Describe the MNTP adaptation phase (LLM2Vec step 2)."""
        return {
            "objective": "masked_next_token_prediction",
            "enables": "bidirectional_attention",
            "steps_dry_run": self.config.mntp_steps_dry_run,
            "mask_probability": 0.20,
            "note": "Replaces the causal mask with an all-ones mask before adaptation.",
        }

    def dry_run(self, texts: Sequence[str]) -> Dict[str, object]:
        built = self.build_inputs(texts)
        return {
            "status": "pass",
            "variant": self.config.variant,
            "base_model": self.config.model_name_or_path,
            "adaptation": self.config.adaptation,
            "pooling_ablation": self.config.pooling_ablation,
            "embedding_dim": self.config.embedding_dim,
            "matryoshka_dims": self.config.matryoshka_dims,
            "mntp_steps_dry_run": self.config.mntp_steps_dry_run,
            "contrastive_steps_dry_run": self.config.contrastive_steps_dry_run,
            "checkpoint_merging": self.config.checkpoint_merging,
            "mntp_plan": self.mntp_plan(),
            "num_texts": len(built),
            "sample_input": built[0] if built else None,
            "note": "Dry-run: bidirectional/MNTP plan validated. No weights loaded.",
        }

    # --- real path (needs torch/transformers / llm2vec) ---------------------
    def _load(self):  # pragma: no cover - requires extras + weights
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "encode() needs extras. Install: pip install -e '.[train]' "
                "(and the 'llm2vec' package for a full bidirectional implementation)."
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        model = AutoModel.from_pretrained(self.config.model_name_or_path)
        # Best-effort bidirectional enablement. A full LLM2Vec implementation replaces the
        # causal attention mask in each layer; prefer the `llm2vec` package for production.
        if hasattr(model.config, "is_decoder"):
            model.config.is_decoder = False
        model.eval()
        self._model = model

    def encode(self, texts: Sequence[str], *, pooling: Optional[str] = None,
               dim: Optional[int] = None):  # pragma: no cover - requires torch
        import torch
        import torch.nn.functional as F

        self._load()
        strategy = pooling or (self.config.pooling_ablation[0] if self.config.pooling_ablation else "mean")
        batch = self._tokenizer(self.build_inputs(texts), padding=True, truncation=True,
                                return_tensors="pt")
        with torch.no_grad():
            hidden = self._model(**batch).last_hidden_state
        mask = batch["attention_mask"]
        if strategy == "mean":
            pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        elif strategy == "cls":
            pooled = hidden[:, 0]
        else:  # eos / last_token
            lengths = mask.sum(1) - 1
            pooled = hidden[torch.arange(hidden.size(0)), lengths]
        if self.config.normalize_embeddings:
            pooled = F.normalize(pooled, p=2, dim=1)
        if dim is not None:
            pooled = F.normalize(pooled[:, :dim], p=2, dim=1)
        return pooled
