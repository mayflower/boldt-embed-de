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
from .metrics import accuracy, aggregate, metrics_for_query, spearman, v_measure
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


# --------------------------------------------------------- task evals (pluggable encoder)
# `encode` is any callable: List[str] -> List[List[float]] (HashingEncoder().encode, or a
# lambda wrapping a real model's encode_texts).

def retrieval_with_encoder(data: dict, encode, ks: Sequence[int] = (1, 5, 10)) -> dict:
    corpus = data["corpus"]
    doc_ids = [d["id"] for d in corpus]
    doc_vecs = encode([d["text"] for d in corpus])
    q_vecs = encode([q["query"] for q in data["queries"]])
    rows = []
    for i, q in enumerate(data["queries"]):
        ranked = cosine_rank(q_vecs[i], list(zip(doc_ids, doc_vecs)))
        rows.append(metrics_for_query(ranked, set(q["positive_doc_ids"]), ks))
    return aggregate(rows)


def evaluate_sts(pairs: Sequence[dict], encode) -> dict:
    a = encode([p["a"] for p in pairs])
    b = encode([p["b"] for p in pairs])
    sims = [cosine_similarity(a[i], b[i]) for i in range(len(pairs))]
    gold = [p["score"] for p in pairs]
    return {"spearman": spearman(sims, gold), "n": len(pairs)}


def evaluate_classification(train_items: Sequence[dict], test_items: Sequence[dict], encode) -> dict:
    tr = encode([it["text"] for it in train_items])
    groups: Dict[str, List[List[float]]] = defaultdict(list)
    for it, v in zip(train_items, tr):
        groups[it["label"]].append(v)
    centroids = {lab: [sum(c[j] for c in vs) / len(vs) for j in range(len(vs[0]))]
                 for lab, vs in groups.items()}
    te = encode([it["text"] for it in test_items])
    preds = []
    for v in te:
        best_lab, best_sim = None, None
        for lab, c in centroids.items():
            s = cosine_similarity(v, c)
            if best_sim is None or s > best_sim:
                best_sim, best_lab = s, lab
        preds.append(best_lab)
    return {"accuracy": accuracy([it["label"] for it in test_items], preds),
            "n_test": len(test_items), "n_classes": len(centroids)}


def _kmeans(vectors: Sequence[Sequence[float]], k: int, iters: int = 15) -> List[int]:
    centroids = [list(vectors[i]) for i in range(min(k, len(vectors)))]
    labels = [0] * len(vectors)
    for _ in range(iters):
        for i, v in enumerate(vectors):
            best, best_d = 0, None
            for ci, c in enumerate(centroids):
                d = sum((a - b) ** 2 for a, b in zip(v, c))
                if best_d is None or d < best_d:
                    best_d, best = d, ci
            labels[i] = best
        new = []
        for ci in range(len(centroids)):
            members = [vectors[i] for i in range(len(vectors)) if labels[i] == ci]
            if members:
                dim = len(members[0])
                new.append([sum(m[j] for m in members) / len(members) for j in range(dim)])
            else:
                new.append(centroids[ci])
        centroids = new
    return labels


def evaluate_clustering(items: Sequence[dict], encode, k: int) -> dict:
    vecs = encode([it["text"] for it in items])
    pred = _kmeans(vecs, k)
    true = [it["label"] for it in items]
    return {"v_measure": v_measure(true, pred), "n": len(items), "k": k}


def evaluate_stress(data: dict, ks: Sequence[int] = (1, 3, 10)) -> dict:
    """Per-category German stress retrieval using the BM25 lexical baseline (deterministic)."""
    corpus = data["corpus"]
    by_cat: Dict[str, List[dict]] = defaultdict(list)
    overall = []
    for c in data["cases"]:
        ranked = bm25_rank(c["query"], corpus)
        m = metrics_for_query(ranked, set(c["positive_doc_ids"]), ks)
        by_cat[c["case"]].append(m)
        overall.append(m)
    return {"by_case": {k: aggregate(v) for k, v in sorted(by_cat.items())},
            "overall": aggregate(overall)}


def efficiency_report(encoder_dim: int, matryoshka_dims: Sequence[int]) -> dict:
    """Storage efficiency of Matryoshka truncation (fp32 bytes/vector per dim)."""
    return {
        "bytes_per_vector_full_fp32": 4 * encoder_dim,
        "by_dim": {
            d: {"bytes": 4 * d, "fraction_of_full": round(d / encoder_dim, 4)}
            for d in matryoshka_dims if d <= encoder_dim
        },
    }
