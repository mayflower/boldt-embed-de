"""Boldt-Embed-DE: a German-first embedding model family based on Boldt/Boldt-DC-350M.

Variants
--------
- ``Boldt-Embed-DE-350M-v1-causal``  : causal decoder embedder, EOS/last-token pooling.
- ``Boldt-Embed-DE-350M-v1-bi``      : bidirectional (LLM2Vec/MNTP-style) adapted embedder.
- ``Boldt-Reranker-DE-350M-v1``      : German cross-encoder reranker.

Design rule
-----------
The importable core of this package (config, pooling, matryoshka, metrics, losses,
data, hard_negatives, eval_harness) depends ONLY on the Python standard library so
that every validation gate runs without GPUs, model weights, or third-party wheels.

Modules that need ``torch``/``transformers`` (the model wrappers and trainers) import
those libraries lazily, inside functions, and raise a clear error if they are missing.
"""

__version__ = "0.1.0"

VARIANTS = (
    "Boldt-Embed-DE-350M-v1-causal",
    "Boldt-Embed-DE-350M-v1-bi",
    "Boldt-Reranker-DE-350M-v1",
)

BASE_MODEL = "Boldt/Boldt-DC-350M"
