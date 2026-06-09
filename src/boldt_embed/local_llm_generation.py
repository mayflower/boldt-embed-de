"""Interface stub for local-LLM synthetic query generation (future work).

Template-based generation (:mod:`boldt_embed.synthetic_queries`) is the current path. This
module defines the *interface* a local generative model would implement so the rest of the
pipeline can switch to it without changes — but it intentionally has no implementation and
makes **no external API / network calls**. Calling it raises ``NotImplementedError``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

_NOT_IMPLEMENTED = (
    "Local-LLM query generation is not implemented. Use the template-based generator "
    "(boldt_embed.synthetic_queries.generate_synthetic_candidates) and filter with the "
    "teacher cache. A local model (e.g. a German instruct LLM via vLLM/transformers) can "
    "implement LocalLLMGenerator.generate_queries_with_local_model in a future change."
)


class LocalLLMGenerator:
    """Placeholder for a local generative model that produces queries from passages.

    A real implementation would load a German instruction-tuned model (no remote API) and
    return query strings; downstream code already handles the candidate-row construction.
    """

    def __init__(self, model_name: Optional[str] = None, device: str = "cuda") -> None:
        self.model_name = model_name
        self.device = device

    def generate_queries_with_local_model(self, passage: Dict[str, Any],
                                          n: int = 4,
                                          styles: Optional[Sequence[str]] = None
                                          ) -> List[str]:
        raise NotImplementedError(_NOT_IMPLEMENTED)


def generate_queries_with_local_model(*args: Any, **kwargs: Any) -> List[str]:
    """Module-level convenience wrapper — also unimplemented (no network calls)."""
    raise NotImplementedError(_NOT_IMPLEMENTED)
