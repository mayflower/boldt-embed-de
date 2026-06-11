"""Teacher scoring pipeline for the 2026 distillation workflow (lazy ML imports).

Two clearly separated layers:

* **stdlib layer** — candidate/cache schema, JSONL read/write, resume dedup, dry-run
  planning. Pure standard library; safe to import in unit tests and behind ``--dry-run``.
* **ML layer** — ``load_*_teacher`` / ``encode_*`` / ``score_*`` functions. These import
  ``torch`` / ``sentence_transformers`` *inside the function body only*, so importing this
  module never pulls in ML dependencies. Real teacher inference runs only when these are
  explicitly called (i.e. behind a non-dry-run CLI execution on a GPU).

Teacher model names come from ``configs/teacher_models.json`` (see :mod:`config_teacher`);
nothing here is hard-coded to Qwen, so the teachers are swappable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

SCORE_VERSION = "teacher-cache-v1"

# Minimum fields an *input* candidate row must carry to be scorable.
CANDIDATE_REQUIRED = ("query_id", "doc_id", "query", "document")

# Full ordered key set of an output cache row.
CACHE_FIELDS = (
    "query_id", "doc_id", "query", "document", "label", "source", "domain", "positive",
    "embedding_teacher_model", "embedding_score", "reranker_teacher_model", "reranker_score",
    "score_version", "created_at",
)

CacheKey = Tuple[str, str]


# ----------------------------------------------------------------- stdlib: schema/IO
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_candidate_record(row: Any) -> List[str]:
    """Return a list of problems with an input candidate row (never raises)."""
    errors: List[str] = []
    if not isinstance(row, dict):
        return ["candidate must be a JSON object"]
    for key in CANDIDATE_REQUIRED:
        val = row.get(key)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"missing/empty required field '{key}'")
    if "positive" in row and row["positive"] is not None and not isinstance(row["positive"], bool):
        errors.append("'positive' must be a bool or null")
    if "label" in row and row["label"] is not None and not isinstance(row["label"], (int, float)):
        errors.append("'label' must be a number or null")
    return errors


def cache_key(row: Dict[str, Any]) -> CacheKey:
    return (str(row.get("query_id")), str(row.get("doc_id")))


def stream_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Yield non-empty JSON objects from a JSONL file."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_candidates(path: str | Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(stream_jsonl(path)):
        if limit is not None and i >= limit:
            break
        out.append(row)
    return out


def read_teacher_cache_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return list(stream_jsonl(p))


def existing_cache_keys(path: str | Path) -> set:
    """(query_id, doc_id) pairs already scored — used to resume without rescoring."""
    keys = set()
    p = Path(path)
    if not p.exists():
        return keys
    for row in stream_jsonl(p):
        keys.add(cache_key(row))
    return keys


def write_teacher_cache_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]],
                              append: bool = False) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    n = 0
    with p.open(mode, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def make_cache_row(candidate: Dict[str, Any], *, embedding_teacher_model: Optional[str] = None,
                   embedding_score: Optional[float] = None,
                   reranker_teacher_model: Optional[str] = None,
                   reranker_score: Optional[float] = None,
                   created_at: Optional[str] = None) -> Dict[str, Any]:
    """Build a fully-formed cache row from an input candidate + optional teacher scores."""
    return {
        "query_id": candidate["query_id"],
        "doc_id": candidate["doc_id"],
        "query": candidate["query"],
        "document": candidate["document"],
        "label": candidate.get("label"),
        "source": candidate.get("source"),
        "domain": candidate.get("domain"),
        "positive": candidate.get("positive"),
        "embedding_teacher_model": embedding_teacher_model,
        "embedding_score": embedding_score,
        "reranker_teacher_model": reranker_teacher_model,
        "reranker_score": reranker_score,
        "score_version": SCORE_VERSION,
        "created_at": created_at or _now_iso(),
    }


def shard_candidates(candidates: Sequence[Dict[str, Any]], shard_size: int
                     ) -> List[List[Dict[str, Any]]]:
    """Split candidates into shards of at most ``shard_size`` (deterministic, order-preserving)."""
    if shard_size <= 0:
        return [list(candidates)]
    return [list(candidates[i:i + shard_size]) for i in range(0, len(candidates), shard_size)]


def shard_path(out_dir: str | Path, prefix: str, index: int) -> str:
    return str(Path(out_dir) / f"{prefix}.shard-{index:05d}.jsonl")


def _score_distribution(values: Sequence[Any]) -> Dict[str, Any]:
    import statistics
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return {"n": 0, "missing": len(values)}
    return {"n": len(nums), "missing": len(values) - len(nums),
            "min": round(min(nums), 4), "median": round(statistics.median(nums), 4),
            "mean": round(statistics.mean(nums), 4), "max": round(max(nums), 4)}


def summarize_cache(rows: Sequence[Dict[str, Any]], suspicious_low_n: int = 5) -> Dict[str, Any]:
    """Quality report over a teacher cache (pure stdlib): counts by source/domain/license, score
    distributions, missing-score counts, suspicious low-scoring positives, near-duplicate counts."""
    def _counts(key):
        c: Dict[str, int] = {}
        for r in rows:
            v = str(r.get(key) or (r.get("metadata") or {}).get(key) or "unknown")
            c[v] = c.get(v, 0) + 1
        return dict(sorted(c.items()))

    positives = [r for r in rows if r.get("positive") is True]
    pos_rr = [(r.get("reranker_score"), r) for r in positives if r.get("reranker_score") is not None]
    pos_rr.sort(key=lambda kv: kv[0])
    pair_hashes = [r.get("pair_hash") for r in rows if r.get("pair_hash")]
    return {
        "total_rows": len(rows),
        "positives": len(positives),
        "by_source": _counts("source"),
        "by_domain": _counts("domain"),
        "by_license": _counts("license"),
        "embedding_score": _score_distribution([r.get("embedding_score") for r in rows]),
        "reranker_score": _score_distribution([r.get("reranker_score") for r in rows]),
        "missing_embedding_score": sum(1 for r in rows if r.get("embedding_score") is None),
        "missing_reranker_score": sum(1 for r in rows if r.get("reranker_score") is None),
        "suspicious_low_positives": [
            {"query": r.get("query", "")[:80], "reranker_score": s}
            for s, r in pos_rr[:suspicious_low_n]],
        "near_duplicate_pairs": len(pair_hashes) - len(set(pair_hashes)),
    }


def filter_cache(rows: Sequence[Dict[str, Any]], reranker_threshold: float
                 ) -> Dict[str, List[Dict[str, Any]]]:
    """Split into training-keep vs review: positive rows whose reranker score is below threshold
    go to 'review' (reason recorded); negatives and high-scoring positives are kept. Adds a
    ``filtering_reason`` field. Pure stdlib."""
    kept, review = [], []
    for r in rows:
        out = dict(r)
        if r.get("positive") is True:
            s = r.get("reranker_score")
            if s is None:
                out["filtering_reason"] = "kept_positive_no_reranker_score"
                kept.append(out)
            elif float(s) >= reranker_threshold:
                out["filtering_reason"] = None
                kept.append(out)
            else:
                out["filtering_reason"] = f"positive_below_reranker_threshold_{reranker_threshold}"
                review.append(out)
        else:
            out["filtering_reason"] = None
            kept.append(out)
    return {"kept": kept, "review": review}


def plan_preview_rows(candidates: Sequence[Dict[str, Any]], mode: str,
                      embedding_model: Optional[str], reranker_model: Optional[str],
                      n: int = 3) -> List[Dict[str, Any]]:
    """Skeleton cache rows (scores left null) for dry-run preview — no ML imports."""
    emb = embedding_model if mode in ("embedding", "both") else None
    rr = reranker_model if mode in ("reranker", "both") else None
    rows = []
    for cand in candidates[:n]:
        rows.append(make_cache_row(cand, embedding_teacher_model=emb,
                                   reranker_teacher_model=rr, created_at="<dry-run>"))
    return rows


def filter_unscored(candidates: Sequence[Dict[str, Any]], done: set) -> List[Dict[str, Any]]:
    """Drop candidates whose (query_id, doc_id) is already present in the cache."""
    return [c for c in candidates if cache_key(c) not in done]


# --------------------------------------------------------------- ML layer (lazy import)
def _resolve_dtype(name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
        name, torch.bfloat16)


def load_embedding_teacher(cfg, device: Optional[str] = None):
    """Load the embedding teacher (SentenceTransformer). Falls back to eager attention
    if flash-attention-2 is requested but unavailable."""
    from sentence_transformers import SentenceTransformer

    device = device or cfg.device
    model_kwargs: Dict[str, Any] = {"torch_dtype": _resolve_dtype(cfg.torch_dtype)}
    if cfg.use_flash_attention_2_if_available:
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            pass  # eager attention; documented fallback
    try:
        st = SentenceTransformer(cfg.model_name, device=device, model_kwargs=model_kwargs)
    except Exception:
        model_kwargs.pop("attn_implementation", None)
        st = SentenceTransformer(cfg.model_name, device=device, model_kwargs=model_kwargs)
    # Cap sequence length to the configured max — large 8B teachers OOM on long inputs at
    # their native (32k) max_seq_length, especially on a shared GPU.
    try:
        st.max_seq_length = int(cfg.max_length)
    except Exception:
        pass
    return st


def load_reranker_teacher(cfg, device: Optional[str] = None):
    """Load the reranker teacher (CrossEncoder)."""
    from sentence_transformers import CrossEncoder

    device = device or cfg.device
    model_kwargs: Dict[str, Any] = {"torch_dtype": _resolve_dtype(cfg.torch_dtype)}
    try:
        return CrossEncoder(cfg.model_name, device=device, max_length=cfg.max_length,
                            automodel_args=model_kwargs)
    except TypeError:
        # Older/newer sentence-transformers signatures differ; fall back to the minimal call.
        return CrossEncoder(cfg.model_name, device=device, max_length=cfg.max_length)


def _instruct_query(instruction: Optional[str], query: str) -> str:
    if instruction:
        return f"Instruct: {instruction}\nQuery: {query}"
    return query


def encode_queries(model, queries: Sequence[str], cfg, batch_size: Optional[int] = None):
    texts = [_instruct_query(cfg.query_instruction, q) for q in queries]
    return model.encode(texts, batch_size=batch_size or cfg.batch_size,
                        normalize_embeddings=cfg.normalize, convert_to_tensor=True,
                        show_progress_bar=False)


def encode_documents(model, documents: Sequence[str], cfg, batch_size: Optional[int] = None):
    if cfg.document_instruction:
        documents = [f"{cfg.document_instruction}\n{d}" for d in documents]
    return model.encode(list(documents), batch_size=batch_size or cfg.batch_size,
                        normalize_embeddings=cfg.normalize, convert_to_tensor=True,
                        show_progress_bar=False)


def score_pairs_with_embedding_teacher(model, pairs: Sequence[Tuple[str, str]], cfg,
                                       batch_size: Optional[int] = None) -> List[float]:
    """Cosine similarity per (query, document) pair from the embedding teacher."""
    if not pairs:
        return []
    queries = [q for q, _ in pairs]
    docs = [d for _, d in pairs]
    q_emb = encode_queries(model, queries, cfg, batch_size)
    d_emb = encode_documents(model, docs, cfg, batch_size)
    sims = (q_emb * d_emb).sum(dim=1)  # both normalized -> cosine
    return [float(x) for x in sims.tolist()]


def score_pairs_with_reranker_teacher(model, pairs: Sequence[Tuple[str, str]], cfg,
                                      batch_size: Optional[int] = None) -> List[float]:
    """Reranker relevance score per (query, document) pair."""
    if not pairs:
        return []
    instr = getattr(cfg, "instruction", None)
    formatted = [(_instruct_query(instr, q), d) for q, d in pairs]
    scores = model.predict(formatted, batch_size=batch_size or cfg.batch_size,
                           show_progress_bar=False)
    if cfg.score_activation == "sigmoid":
        import torch
        scores = torch.sigmoid(torch.as_tensor(scores)).tolist()
    return [float(x) for x in scores]


def score_candidates_for_queries(candidates: Sequence[Dict[str, Any]], teacher_cfg, mode: str,
                                 *, embedding_model=None, reranker_model=None,
                                 batch_size_embedding: Optional[int] = None,
                                 batch_size_reranker: Optional[int] = None,
                                 device: Optional[str] = None) -> List[Dict[str, Any]]:
    """Score every candidate (query, document) pair with the requested teacher(s) and
    return fully-formed cache rows. Loads any teacher not passed in. ML-only."""
    pairs = [(c["query"], c["document"]) for c in candidates]
    emb_scores: List[Optional[float]] = [None] * len(candidates)
    rr_scores: List[Optional[float]] = [None] * len(candidates)
    emb_name = rr_name = None

    if mode in ("embedding", "both"):
        emb_cfg = teacher_cfg.embedding_teacher
        emb_name = emb_cfg.model_name
        model = embedding_model or load_embedding_teacher(emb_cfg, device)
        emb_scores = score_pairs_with_embedding_teacher(model, pairs, emb_cfg,
                                                        batch_size_embedding)
        if embedding_model is None and mode == "both":
            # Free the 8B embedding teacher before loading the 8B reranker — two at once
            # will not fit on a shared 48GB GPU.
            import gc
            import torch
            del model
            gc.collect()
            torch.cuda.empty_cache()
    if mode in ("reranker", "both"):
        rr_cfg = teacher_cfg.reranker_teacher
        rr_name = rr_cfg.model_name
        model = reranker_model or load_reranker_teacher(rr_cfg, device)
        rr_scores = score_pairs_with_reranker_teacher(model, pairs, rr_cfg, batch_size_reranker)

    created = _now_iso()
    rows = []
    for cand, es, rs in zip(candidates, emb_scores, rr_scores):
        rows.append(make_cache_row(
            cand, embedding_teacher_model=emb_name, embedding_score=es,
            reranker_teacher_model=rr_name, reranker_score=rs, created_at=created))
    return rows
