"""Scalable lexical (BM25) index built ONCE over a corpus, then queried many times (stdlib).

The v2 mining bottleneck: ``eval_harness.bm25_rank`` re-tokenizes the *entire* corpus on every
query, so ``mine_bm25_candidates`` was O(n_queries * n_corpus) and mining was capped to a ~3.5k
subset. This module builds an inverted index up front; each search then touches only the
postings of the query terms, so a full-corpus scan over 100k+ docs is feasible.

German handling: lowercase + ``ß``→``ss`` always (standard fold); umlaut folding
(``ä``→``ae`` …) is optional (``fold_umlauts``) and recorded in the index so query-time
tokenization matches index-time. No third-party deps, no ML, no network.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_TOKEN_RE = re.compile(r"[0-9a-zäöü]+")
_UMLAUT_MAP = {"ä": "ae", "ö": "oe", "ü": "ue"}


def tokenize_de(text: Any, fold_umlauts: bool = False) -> List[str]:
    """Lowercase, fold ``ß``→``ss`` (always), optionally fold umlauts, then split to tokens."""
    t = str(text).lower().replace("ß", "ss")
    if fold_umlauts:
        t = t.translate(str.maketrans(_UMLAUT_MAP))
    return _TOKEN_RE.findall(t)


class BM25Index:
    """Okapi BM25 over an inverted index. Document ids are preserved and returned by search."""

    FORMAT = "bm25-index-v1"
    _BUILD_COUNT = 0  # class-level instrumentation: how many times build() ran (tests assert once)

    def __init__(self, k1: float = 1.5, b: float = 0.75, fold_umlauts: bool = False):
        self.k1 = k1
        self.b = b
        self.fold_umlauts = fold_umlauts
        self.doc_ids: List[str] = []
        self.doc_len: List[int] = []
        self.postings: Dict[str, List[Tuple[int, int]]] = {}   # term -> [(doc_idx, tf), ...]
        self.df: Dict[str, int] = {}
        self.n_docs = 0
        self.avgdl = 0.0
        self._idf: Dict[str, float] = {}

    # -------------------------------------------------------------------------- build
    def build(self, documents: Iterable[Any], text_field: str = "text",
              id_field: str = "doc_id") -> "BM25Index":
        """Index ``documents`` (dicts with id/text fields, or (id, text) tuples). Build once."""
        BM25Index._BUILD_COUNT += 1
        self.doc_ids, self.doc_len = [], []
        self.postings, self.df = {}, {}
        for doc in documents:
            if isinstance(doc, (tuple, list)):
                did, text = str(doc[0]), doc[1]
            else:
                did = str(doc.get(id_field) or doc.get("id") or doc.get("doc_id") or len(self.doc_ids))
                text = doc.get(text_field) or doc.get("text") or doc.get("document") or ""
            idx = len(self.doc_ids)
            self.doc_ids.append(did)
            toks = tokenize_de(text, self.fold_umlauts)
            self.doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings.setdefault(term, []).append((idx, tf))
                self.df[term] = self.df.get(term, 0) + 1
        self.n_docs = len(self.doc_ids)
        total = sum(self.doc_len)
        self.avgdl = (total / self.n_docs) if self.n_docs else 0.0
        self._idf = {t: math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
                     for t, df in self.df.items()}
        return self

    # ------------------------------------------------------------------------- search
    def _score_query(self, query: Any) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        for term in set(tokenize_de(query, self.fold_umlauts)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for doc_idx, tf in postings:
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[doc_idx] / (self.avgdl or 1.0))
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * (tf * (self.k1 + 1)) / denom
        return scores

    def search(self, query: Any, top_k: int = 50) -> List[Tuple[str, float]]:
        """Top-k (doc_id, score), score desc with doc_id asc as a deterministic tie-break."""
        scores = self._score_query(query)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.doc_ids[kv[0]]))
        return [(self.doc_ids[i], round(s, 6)) for i, s in ranked[:top_k]]

    def batch_search(self, queries: Sequence[Any], top_k: int = 50) -> List[List[Tuple[str, float]]]:
        """Search many queries against the already-built index (no rebuild). Equals per-query search."""
        return [self.search(q, top_k) for q in queries]

    # -------------------------------------------------------------------- (de)serialize
    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.FORMAT, "k1": self.k1, "b": self.b, "fold_umlauts": self.fold_umlauts,
            "doc_ids": self.doc_ids, "doc_len": self.doc_len, "df": self.df,
            "postings": {t: [list(p) for p in plist] for t, plist in self.postings.items()},
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BM25Index":
        idx = cls(k1=float(d.get("k1", 1.5)), b=float(d.get("b", 0.75)),
                  fold_umlauts=bool(d.get("fold_umlauts", False)))
        idx.doc_ids = list(d["doc_ids"])
        idx.doc_len = list(d["doc_len"])
        idx.df = {t: int(v) for t, v in d["df"].items()}
        idx.postings = {t: [(int(i), int(tf)) for i, tf in plist]
                        for t, plist in d["postings"].items()}
        idx.n_docs = len(idx.doc_ids)
        total = sum(idx.doc_len)
        idx.avgdl = (total / idx.n_docs) if idx.n_docs else 0.0
        idx._idf = {t: math.log(1 + (idx.n_docs - df + 0.5) / (df + 0.5)) for t, df in idx.df.items()}
        return idx

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def build_bm25_index(documents: Iterable[Any], text_field: str = "text",
                     id_field: str = "doc_id", fold_umlauts: bool = False) -> BM25Index:
    return BM25Index(fold_umlauts=fold_umlauts).build(documents, text_field, id_field)
