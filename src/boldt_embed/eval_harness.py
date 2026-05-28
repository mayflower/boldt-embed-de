"""Retrieval evaluation harness (pure stdlib).

Provides two no-dependency retrievers so the metric/Matryoshka plumbing runs end-to-end
without weights:

- ``bm25_rank``  : a lexical BM25 baseline.
- ``HashingEncoder`` : a DETERMINISTIC char-n-gram hashing encoder used ONLY as a
  plumbing stand-in for a real embedder. It is NOT the Boldt model and its scores are
  NOT a quality claim — swap in ``CausalEmbedder``/``BidirectionalEmbedder.encode`` for
  real evaluation.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .losses import cosine_similarity
from .matryoshka import truncate_normalized
from .metrics import aggregate, metrics_for_query
from .textutil import normalize, tokenize


# --------------------------------------------------------------------------- BM25
def bm25_rank(query: str, corpus: Sequence[dict], k1: float = 1.5, b: float = 0.75) -> List[str]:
    doc_tokens = {d["id"]: tokenize(d["text"]) for d in corpus}
    doc_len = {i: len(t) for i, t in doc_tokens.items()}
    avg_len = sum(doc_len.values()) / max(len(doc_len), 1)
    df: Dict[str, int] = defaultdict(int)
    for tokens in doc_tokens.values():
        for tok in set(tokens):
            df[tok] += 1
    n_docs = len(corpus)
    q_tokens = tokenize(query)
    scores: List[Tuple[str, float]] = []
    for d in corpus:
        i = d["id"]
        tf = Counter(doc_tokens[i])
        score = 0.0
        for tok in q_tokens:
            if tok not in tf:
                continue
            idf = math.log(1 + (n_docs - df[tok] + 0.5) / (df[tok] + 0.5))
            denom = tf[tok] + k1 * (1 - b + b * doc_len[i] / avg_len)
            score += idf * (tf[tok] * (k1 + 1)) / denom
        scores.append((i, score))
    scores.sort(key=lambda kv: kv[1], reverse=True)
    return [i for i, _ in scores]


# ------------------------------------------------------------------- HashingEncoder
class HashingEncoder:
    """Deterministic char-n-gram hashing encoder (plumbing stand-in, NOT Boldt)."""

    def __init__(self, dim: int = 256, ngram: int = 3) -> None:
        self.dim = dim
        self.ngram = ngram

    def _vector(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        norm = normalize(text)
        grams = [norm[i : i + self.ngram] for i in range(len(norm) - self.ngram + 1)] or [norm]
        for gram in grams:
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm_val = math.sqrt(sum(x * x for x in vec))
        return [x / norm_val for x in vec] if norm_val > 0 else vec

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._vector(t) for t in texts]


def cosine_rank(query_vec: Sequence[float], doc_vecs: Sequence[Tuple[str, Sequence[float]]]) -> List[str]:
    scored = [(i, cosine_similarity(query_vec, v)) for i, v in doc_vecs]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return [i for i, _ in scored]


# ------------------------------------------------------------------------- runners
def run_retrieval(data: dict, rank_fn: Callable[[str], List[str]], ks: Sequence[int]) -> dict:
    rows = []
    per_query = []
    for q in data["queries"]:
        ranked = rank_fn(q["query"])
        positives = set(q["positive_doc_ids"])
        m = metrics_for_query(ranked, positives, ks)
        rows.append(m)
        per_query.append({
            "query_id": q["id"], "query": q["query"],
            "positive_doc_ids": sorted(positives), "top5": ranked[:5], "metrics": m,
        })
    return {"aggregate": aggregate(rows), "queries": per_query}


def evaluate_bm25(data: dict, ks: Sequence[int] = (1, 3, 5, 10)) -> dict:
    corpus = data["corpus"]
    return run_retrieval(data, lambda q: bm25_rank(q, corpus), ks)


def evaluate_hashing(
    data: dict,
    ks: Sequence[int] = (1, 3, 5, 10),
    matryoshka_dims: Optional[Sequence[int]] = None,
    dim: int = 256,
    ngram: int = 3,
) -> dict:
    enc = HashingEncoder(dim=dim, ngram=ngram)
    corpus = data["corpus"]
    doc_ids = [d["id"] for d in corpus]
    doc_vecs = enc.encode([d["text"] for d in corpus])

    def rank_full(query: str) -> List[str]:
        qv = enc.encode([query])[0]
        return cosine_rank(qv, list(zip(doc_ids, doc_vecs)))

    result = {"full": run_retrieval(data, rank_full, ks)}
    if matryoshka_dims:
        by_dim: Dict[int, dict] = {}
        for d_dim in matryoshka_dims:
            if d_dim > dim:
                continue
            d_vecs = [truncate_normalized(v, d_dim) for v in doc_vecs]

            def rank_dim(query: str, dd: int = d_dim, dv=d_vecs) -> List[str]:
                qv = truncate_normalized(enc.encode([query])[0], dd)
                return cosine_rank(qv, list(zip(doc_ids, dv)))

            by_dim[d_dim] = run_retrieval(data, rank_dim, ks)["aggregate"]
        result["by_dim"] = by_dim
    return result


def summarize_stress(cases: Sequence[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for c in cases:
        counts[c.get("case", "unknown")] += 1
    return dict(sorted(counts.items()))
