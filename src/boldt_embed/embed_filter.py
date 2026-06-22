"""EmbedFilter — spectral-bulk projection from a model's unembedding matrix (arXiv 2606.07502).

A dimensionality-reduction **postprocessor**: project a pooled embedding onto the centered "bulk"
slice of the unembedding matrix's right singular vectors (``Vh``), then L2-normalize. It is an
alternative to prefix Matryoshka truncation (`matryoshka.truncate_normalized`) and is meant to be
compared head-to-head against it at equal output dims.

This module is the **stdlib-only core** — spec selection + metadata validation. It imports NO
``torch``. The GPU basis build lives in ``scripts/build_embed_filter.py``; the lazy-torch projection
loader is :func:`load_embed_filter_basis` below (``torch`` is imported *inside* the function only).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

ALLOWED_TAUS: Tuple[int, ...] = (1, 2, 4, 8, 16)
SOURCE_MATRICES = ("output_embeddings", "lm_head", "tied_input_embeddings")


@dataclass
class EmbedFilterSpec:
    """A centered "bulk" slice of the unembedding right-singular directions."""
    hidden_dim: int
    tau: int
    keep_dim: int
    left: int
    right: int
    strategy: str = "bulk_center"


def _is_pos_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def select_bulk_slice(hidden_dim: int, tau: int) -> EmbedFilterSpec:
    """Pick the centered slice of ``K = hidden_dim // tau`` singular directions.

    ``tau=1`` keeps everything (left=0, right=hidden_dim). Raises ``ValueError`` on an unsupported
    tau, a non-positive hidden_dim, or a non-divisible (hidden_dim, tau) pair.
    """
    if not _is_pos_int(hidden_dim):
        raise ValueError(f"hidden_dim must be a positive int, got {hidden_dim!r}")
    if tau not in ALLOWED_TAUS:
        raise ValueError(f"tau must be one of {ALLOWED_TAUS}, got {tau!r}")
    if hidden_dim % tau != 0:
        raise ValueError(f"hidden_dim {hidden_dim} is not divisible by tau {tau}")
    keep = hidden_dim // tau
    left = (hidden_dim - keep) // 2
    right = left + keep
    return EmbedFilterSpec(hidden_dim=hidden_dim, tau=tau, keep_dim=keep, left=left, right=right)


def metadata_for_spec(spec: EmbedFilterSpec, *, model: str, source_matrix: str,
                      vocab_size: Optional[int] = None,
                      extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """JSON-serializable metadata for run cards and the ``metadata.json`` artifact sidecar."""
    meta: Dict[str, Any] = {
        "kind": "embed_filter",
        "model": model,
        "hidden_dim": spec.hidden_dim,
        "tau": spec.tau,
        "keep_dim": spec.keep_dim,
        "left": spec.left,
        "right": spec.right,
        "strategy": spec.strategy,
        "source_matrix": source_matrix,
        "vocab_size": vocab_size,
        "artifact_format": "basis.pt:[hidden_dim,keep_dim] + metadata.json",
    }
    if extra:
        meta.update(extra)
    return meta


def validate_embed_filter_metadata(meta: Dict[str, Any]) -> List[str]:
    """Return a list of problems with an EmbedFilter metadata dict (empty == valid). Stdlib only."""
    if not isinstance(meta, dict):
        return ["metadata must be a JSON object"]
    errors: List[str] = []

    if not isinstance(meta.get("model"), str) or not str(meta.get("model")).strip():
        errors.append("'model' must be a non-empty string")

    H, tau, keep = meta.get("hidden_dim"), meta.get("tau"), meta.get("keep_dim")
    left, right = meta.get("left"), meta.get("right")
    if not _is_pos_int(H):
        errors.append("'hidden_dim' must be a positive int")
    if tau not in ALLOWED_TAUS:
        errors.append(f"'tau' must be one of {ALLOWED_TAUS}")
    if not _is_pos_int(keep):
        errors.append("'keep_dim' must be a positive int")

    if all(isinstance(x, int) and not isinstance(x, bool) for x in (H, keep, left, right)):
        if not (0 <= left <= right <= H):
            errors.append(f"bad bounds: need 0<=left({left})<=right({right})<=H({H})")
        if right - left != keep:
            errors.append(f"right-left ({right - left}) must equal keep_dim ({keep})")
        if _is_pos_int(H) and tau in ALLOWED_TAUS and keep != H // tau:
            errors.append(f"keep_dim {keep} must equal hidden_dim//tau ({H // tau})")

    if meta.get("source_matrix") not in SOURCE_MATRICES:
        errors.append(f"'source_matrix' must be one of {SOURCE_MATRICES}")
    fmt = meta.get("artifact_format")
    if not isinstance(fmt, str) or "basis" not in fmt:
        errors.append("'artifact_format' must describe the basis artifact (mention 'basis')")
    return errors


# --------------------------------------------------------------------------- lazy-torch loader
def load_embed_filter_basis(path_or_dir: str, *, expected_hidden_dim: Optional[int] = None):
    """Load a basis tensor + metadata from an artifact directory (``basis.pt`` + ``metadata.json``)
    or a direct ``.pt`` path. ``torch`` is imported HERE, not at module import time.

    Returns ``(basis_tensor[hidden_dim, keep_dim], metadata_dict)``. Raises ``ValueError`` on a
    metadata/shape problem or a hidden-dim mismatch.
    """
    import json
    import pathlib

    import torch  # lazy

    p = pathlib.Path(path_or_dir)
    if p.is_dir():
        basis_path, meta_path = p / "basis.pt", p / "metadata.json"
    else:
        basis_path, meta_path = p, p.with_suffix(".json")
    if not basis_path.exists():
        raise ValueError(f"basis tensor not found: {basis_path}")

    meta: Dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        problems = validate_embed_filter_metadata(meta)
        if problems:
            raise ValueError("invalid embed-filter metadata: " + "; ".join(problems))

    basis = torch.load(basis_path, map_location="cpu")
    if not hasattr(basis, "shape") or basis.ndim != 2:
        raise ValueError(f"basis must be a 2-D tensor, got shape {getattr(basis, 'shape', None)}")
    h = basis.shape[0]
    if expected_hidden_dim is not None and h != expected_hidden_dim:
        raise ValueError(f"basis hidden_dim {h} != expected {expected_hidden_dim}")
    if meta and meta.get("hidden_dim") not in (None, h):
        raise ValueError(f"basis hidden_dim {h} != metadata hidden_dim {meta.get('hidden_dim')}")
    return basis, meta
