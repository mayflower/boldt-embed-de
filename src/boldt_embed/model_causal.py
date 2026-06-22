"""Causal decoder embedder wrapper.

``torch``/``transformers`` are imported lazily inside ``_load``/``encode`` so the wrapper
(and its dry-run) can be exercised without those libraries installed. The dry-run path is
pure stdlib and validates config + instruction wiring; ``encode`` needs the ``train`` extra.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .config import CausalConfig, load_causal_config
from .instructions import format_document, format_query


class CausalEmbedder:
    def __init__(self, config: CausalConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None

    @classmethod
    def from_config(cls, path: str) -> "CausalEmbedder":
        return cls(load_causal_config(path))

    # --- stdlib, no weights -------------------------------------------------
    def build_inputs(
        self, queries: Sequence[str], documents: Sequence[str]
    ) -> Dict[str, List[str]]:
        return {
            "queries": [format_query(self.config.query_instruction, q) for q in queries],
            "documents": [format_document(self.config.document_instruction, d) for d in documents],
        }

    def dry_run(self, queries: Sequence[str], documents: Sequence[str]) -> Dict[str, object]:
        built = self.build_inputs(queries, documents)
        return {
            "status": "pass",
            "variant": self.config.variant,
            "base_model": self.config.model_name_or_path,
            "pooling": self.config.pooling,
            "embedding_dim": self.config.embedding_dim,
            "matryoshka_dims": self.config.matryoshka_dims,
            "normalize_embeddings": self.config.normalize_embeddings,
            "temperature": self.config.temperature,
            "loss": self.config.loss,
            "num_queries": len(built["queries"]),
            "num_documents": len(built["documents"]),
            "sample_query_input": built["queries"][0] if built["queries"] else None,
            "sample_document_input": built["documents"][0] if built["documents"] else None,
            "note": "Dry-run: config + instruction wiring validated. No weights loaded.",
        }

    # --- real path (needs torch/transformers) -------------------------------
    def _load(self):
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only with extras
            raise ImportError(
                "encode() needs the training extras. Install: pip install -e '.[train]'"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModel.from_pretrained(self.config.model_name_or_path)
        self._model.eval()

    def encode(
        self,
        texts: Sequence[str],
        *,
        is_query: bool = False,
        dim: Optional[int] = None,
        max_length: Optional[int] = None,
        embed_filter: Optional[str] = None,
    ):
        # `dim` (prefix Matryoshka) and `embed_filter` (spectral bulk) are competing reduction
        # methods — refuse to silently combine them. Checked before any torch import.
        if dim is not None and embed_filter is not None:
            raise ValueError(
                "pass either `dim` (prefix Matryoshka) or `embed_filter` (spectral bulk), not both"
            )
        import torch  # pragma: no cover - requires torch + weights, not run in CI
        import torch.nn.functional as F

        self._load()
        templated = (
            self.build_inputs(texts, [])["queries"] if is_query
            else [format_document(self.config.document_instruction, t) for t in texts]
        )
        max_len = max_length or (
            self.config.max_query_length if is_query else self.config.max_document_length
        )
        batch = self._tokenizer(
            list(templated), padding=True, truncation=True, max_length=max_len,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = self._model(**batch).last_hidden_state  # [B, T, H]
        mask = batch["attention_mask"]
        pooled = self._pool(hidden, mask)
        if embed_filter is not None:
            from .embed_filter import load_embed_filter_basis
            basis, _ = load_embed_filter_basis(embed_filter, expected_hidden_dim=pooled.shape[1])
            return F.normalize(pooled @ basis.to(pooled.device, pooled.dtype), p=2, dim=1)
        if self.config.normalize_embeddings:
            pooled = F.normalize(pooled, p=2, dim=1)
        if dim is not None:
            pooled = F.normalize(pooled[:, :dim], p=2, dim=1)
        return pooled

    def _pool(self, hidden, mask):  # pragma: no cover - requires torch
        import torch

        strategy = self.config.pooling
        if strategy == "mean":
            summed = (hidden * mask.unsqueeze(-1)).sum(dim=1)
            counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
            return summed / counts
        if strategy == "cls":
            return hidden[:, 0]
        # eos / last_token / eos_or_last_token: last non-pad token
        if getattr(self._tokenizer, "padding_side", "right") == "left":
            return hidden[:, -1]
        lengths = mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0)), lengths]
