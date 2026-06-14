"""Scalable leakage detection: train candidates vs held-out eval corpora (pure stdlib).

The v2 scan used :func:`boldt_embed.data.find_leakage`, which is O(n_train * n_eval) exact
Jaccard — so v2 only filtered against a subset of eval texts. This module replaces it with a
two-stage scanner that scales to full-corpus checks (100k candidates vs all eval datasets):

1. **Blocking (cheap, near-linear):** index eval texts by exact hash, normalized hash, SimHash
   bands, and MinHash-LSH bands. For each candidate text we look up only the eval texts that
   share a block — typically a handful — instead of all of them.
2. **Verify (exact, few):** compute exact token-shingle Jaccard only for blocked (candidate,
   eval) pairs, and classify exact / exact-normalized / near-duplicate.

Determinism: hashing is `blake2b` (stable across processes, unlike builtin ``hash``); MinHash
permutations are seeded. No third-party deps, no ML, no network.
"""
from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ----------------------------------------------------------------- tunables / defaults
DEFAULT_SHINGLE_N = 5
DEFAULT_NUM_PERM = 64
MINHASH_BANDS = 16          # 16 bands x 4 rows over 64 perms -> recall down to Jaccard ~0.5
SIMHASH_BANDS = 4           # 4 bands x 16 bits over 64-bit simhash
SIMHASH_BAND_BITS = 16
DEFAULT_JACCARD_THRESHOLD = 0.9
DEFAULT_SIMHASH_MAX_HAMMING = 3
_MERSENNE = (1 << 61) - 1


# ---------------------------------------------------------------------------- features
def normalize_text_for_leakage(text: Any) -> str:
    """Lowercase, NFKC, drop punctuation, collapse whitespace. Keeps German word chars
    (umlauts/ß) so near-identical eval/train text collapses to the same normalized form."""
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


def _stable_hash64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def stable_text_hash(text: Any) -> str:
    return hashlib.blake2b(str(text).encode("utf-8"), digest_size=16).hexdigest()


def token_shingles(text: Any, n: int = DEFAULT_SHINGLE_N) -> Set[str]:
    """Set of word n-gram shingles of the normalized text. Short texts -> a single shingle."""
    toks = normalize_text_for_leakage(text).split()
    if not toks:
        return set()
    if len(toks) < n:
        return {" ".join(toks)}
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def simhash64(text: Any, n: int = DEFAULT_SHINGLE_N) -> int:
    """64-bit SimHash over token shingles. Near-identical texts -> small Hamming distance."""
    feats = token_shingles(text, n)
    if not feats:
        return 0
    bits = [0] * 64
    for f in feats:
        h = _stable_hash64(f)
        for i in range(64):
            bits[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(64):
        if bits[i] > 0:
            out |= (1 << i)
    return out


def _perm_params(num_perm: int, seed: int = 0) -> List[Tuple[int, int]]:
    rnd = random.Random(seed)
    return [(rnd.randrange(1, _MERSENNE), rnd.randrange(0, _MERSENNE)) for _ in range(num_perm)]


_PERMS_CACHE: Dict[int, List[Tuple[int, int]]] = {}


def _perms(num_perm: int) -> List[Tuple[int, int]]:
    if num_perm not in _PERMS_CACHE:
        _PERMS_CACHE[num_perm] = _perm_params(num_perm)
    return _PERMS_CACHE[num_perm]


def minhash_signature(text: Any, num_perm: int = DEFAULT_NUM_PERM,
                      n: int = DEFAULT_SHINGLE_N) -> Tuple[int, ...]:
    """MinHash signature (``num_perm`` mins of affine-permuted shingle hashes). Empty text -> 0s."""
    shingles = token_shingles(text, n)
    if not shingles:
        return tuple([0] * num_perm)
    hs = [_stable_hash64(s) % _MERSENNE for s in shingles]
    sig = []
    for a, b in _perms(num_perm):
        sig.append(min(((a * h + b) % _MERSENNE) for h in hs))
    return tuple(sig)


def hamming64(a: int, b: int) -> int:
    return bin((a ^ b) & ((1 << 64) - 1)).count("1")


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a | b))


def _minhash_bands(sig: Sequence[int], bands: int = MINHASH_BANDS) -> List[int]:
    """Hash each band of the signature to a bucket id. Sharing any bucket = blocking match."""
    rows = max(1, len(sig) // bands)
    out = []
    for bi in range(bands):
        band = tuple(sig[bi * rows:(bi + 1) * rows])
        if band:
            out.append(_stable_hash64(f"{bi}:" + ",".join(map(str, band))))
    return out


def _simhash_bands(sh: int, bands: int = SIMHASH_BANDS, bits: int = SIMHASH_BAND_BITS) -> List[int]:
    out = []
    mask = (1 << bits) - 1
    for bi in range(bands):
        out.append(_stable_hash64(f"{bi}:{(sh >> (bi * bits)) & mask}"))
    return out


# ------------------------------------------------------------------------------- index
@dataclass
class _EvalDoc:
    eval_id: str
    dataset: str
    field: str
    exact_hash: str
    norm_hash: str
    shingles: Set[str]
    simhash: int


@dataclass
class LeakageIndex:
    shingle_n: int = DEFAULT_SHINGLE_N
    num_perm: int = DEFAULT_NUM_PERM
    docs: List[_EvalDoc] = field(default_factory=list)
    by_exact: Dict[str, List[int]] = field(default_factory=dict)
    by_norm: Dict[str, List[int]] = field(default_factory=dict)
    by_minhash_band: Dict[int, List[int]] = field(default_factory=dict)
    by_simhash_band: Dict[int, List[int]] = field(default_factory=dict)

    def n_eval_texts(self) -> int:
        return len(self.docs)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable form (shingles -> sorted lists, int band keys -> str)."""
        return {
            "format": "leakage-index-v1",
            "shingle_n": self.shingle_n, "num_perm": self.num_perm,
            "docs": [{"eval_id": d.eval_id, "dataset": d.dataset, "field": d.field,
                      "exact_hash": d.exact_hash, "norm_hash": d.norm_hash,
                      "shingles": sorted(d.shingles), "simhash": d.simhash} for d in self.docs],
            "by_exact": self.by_exact, "by_norm": self.by_norm,
            "by_minhash_band": {str(k): v for k, v in self.by_minhash_band.items()},
            "by_simhash_band": {str(k): v for k, v in self.by_simhash_band.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LeakageIndex":
        idx = cls(shingle_n=int(d.get("shingle_n", DEFAULT_SHINGLE_N)),
                  num_perm=int(d.get("num_perm", DEFAULT_NUM_PERM)))
        idx.docs = [_EvalDoc(eval_id=x["eval_id"], dataset=x["dataset"], field=x["field"],
                             exact_hash=x["exact_hash"], norm_hash=x["norm_hash"],
                             shingles=set(x["shingles"]), simhash=int(x["simhash"]))
                    for x in d.get("docs", [])]
        idx.by_exact = {k: list(v) for k, v in d.get("by_exact", {}).items()}
        idx.by_norm = {k: list(v) for k, v in d.get("by_norm", {}).items()}
        idx.by_minhash_band = {int(k): list(v) for k, v in d.get("by_minhash_band", {}).items()}
        idx.by_simhash_band = {int(k): list(v) for k, v in d.get("by_simhash_band", {}).items()}
        return idx


_TEXT_FIELDS = ("query", "document", "positive", "text", "context", "title")


def eval_texts_from_record(rec: Dict[str, Any], dataset: str) -> List[Tuple[str, str, str, str]]:
    """Expand one eval row into (eval_id, dataset, field, text) units over all present text fields."""
    rid = str(rec.get("query_id") or rec.get("doc_id") or rec.get("id") or "")
    units = []
    for fld in _TEXT_FIELDS:
        v = rec.get(fld)
        if isinstance(v, str) and v.strip():
            uid = f"{dataset}:{rid or stable_text_hash(v)[:12]}:{fld}"
            units.append((uid, dataset, fld, v))
    return units


def build_eval_leakage_index(eval_records: Iterable[Tuple[str, str, str, str]],
                             shingle_n: int = DEFAULT_SHINGLE_N,
                             num_perm: int = DEFAULT_NUM_PERM) -> LeakageIndex:
    """Build the blocking index. ``eval_records`` = iterable of (eval_id, dataset, field, text)."""
    idx = LeakageIndex(shingle_n=shingle_n, num_perm=num_perm)
    for eval_id, dataset, fld, text in eval_records:
        shingles = token_shingles(text, shingle_n)
        sh = simhash64(text, shingle_n)
        doc = _EvalDoc(eval_id=eval_id, dataset=dataset, field=fld,
                       exact_hash=stable_text_hash(text),
                       norm_hash=stable_text_hash(normalize_text_for_leakage(text)),
                       shingles=shingles, simhash=sh)
        i = len(idx.docs)
        idx.docs.append(doc)
        idx.by_exact.setdefault(doc.exact_hash, []).append(i)
        idx.by_norm.setdefault(doc.norm_hash, []).append(i)
        for band in _minhash_bands(minhash_signature(text, num_perm, shingle_n)):
            idx.by_minhash_band.setdefault(band, []).append(i)
        for band in _simhash_bands(sh):
            idx.by_simhash_band.setdefault(band, []).append(i)
    return idx


# -------------------------------------------------------------------------------- scan
_CAND_FIELDS = ("query", "document")


def _candidate_id(rec: Dict[str, Any]) -> str:
    return str(rec.get("pair_hash") or rec.get("id")
               or f"{rec.get('query_id', '')}|{rec.get('doc_id', '')}".strip("|")
               or stable_text_hash(str(rec.get("query", "")) + str(rec.get("document", "")))[:16])


def find_candidate_leakage(candidate_records: Sequence[Dict[str, Any]], index: LeakageIndex,
                           jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
                           simhash_max_hamming: int = DEFAULT_SIMHASH_MAX_HAMMING
                           ) -> Dict[str, Any]:
    """Two-stage scan. Returns {hits, stats}. A hit is the strongest eval match for a candidate
    text unit (exact > exact_normalized > near_duplicate). ``stats.jaccard_comparisons`` is the
    count of verify-stage exact comparisons — should be << n_train * n_eval (subquadratic)."""
    hits: List[Dict[str, Any]] = []
    comparisons = 0
    blocked_pairs = 0
    n = index.shingle_n
    for rec in candidate_records:
        cid = _candidate_id(rec)
        for fld in _CAND_FIELDS:
            text = rec.get(fld)
            if not (isinstance(text, str) and text.strip()):
                continue
            ex = stable_text_hash(text)
            nh = stable_text_hash(normalize_text_for_leakage(text))
            sh = simhash64(text, n)
            shingles = token_shingles(text, n)

            # ---- stage 1: blocking (gather candidate eval-doc indices) ----
            blocked: Set[int] = set()
            blocked.update(index.by_exact.get(ex, ()))
            blocked.update(index.by_norm.get(nh, ()))
            sig = minhash_signature(text, index.num_perm, n)
            for band in _minhash_bands(sig, MINHASH_BANDS):
                blocked.update(index.by_minhash_band.get(band, ()))
            for band in _simhash_bands(sh):
                blocked.update(index.by_simhash_band.get(band, ()))
            blocked_pairs += len(blocked)

            # ---- stage 2: verify (exact only on blocked) ----
            best: Optional[Dict[str, Any]] = None
            for di in blocked:
                doc = index.docs[di]
                comparisons += 1
                if doc.exact_hash == ex:
                    kind, score = "exact", 1.0
                elif doc.norm_hash == nh:
                    kind, score = "exact_normalized", 1.0
                else:
                    j = jaccard(shingles, doc.shingles)
                    ham = hamming64(sh, doc.simhash)
                    if j >= jaccard_threshold or ham <= simhash_max_hamming:
                        kind, score = "near_duplicate", round(j, 4)
                    else:
                        continue
                rank = {"exact": 3, "exact_normalized": 2, "near_duplicate": 1}[kind]
                if best is None or (rank, score) > (best["_rank"], best["score"]):
                    best = {"candidate_id": cid, "candidate_field": fld, "kind": kind,
                            "score": score, "eval_id": doc.eval_id, "eval_dataset": doc.dataset,
                            "eval_field": doc.field, "_rank": rank,
                            "source": rec.get("source"), "domain": rec.get("domain"),
                            "license": rec.get("license")}
            if best is not None:
                best.pop("_rank", None)
                hits.append(best)
    return {"hits": hits,
            "stats": {"jaccard_comparisons": comparisons, "blocked_pairs": blocked_pairs,
                      "n_candidate_records": len(candidate_records)}}


# --------------------------------------------------------------------- report / gating
def leakage_report_is_clean(report: Dict[str, Any]) -> bool:
    """A report is training-acceptable if it found no leakage OR a cleaned candidate file was
    written (hits dropped)."""
    no_hits = (int(report.get("exact_hits", 0)) + int(report.get("exact_normalized_hits", 0))
               + int(report.get("near_duplicate_hits", 0))) == 0
    return no_hits or bool(report.get("cleaned_candidates_path"))


def require_clean_leakage_report(report_path: str) -> None:
    """Raise unless a full leakage report exists and is clean-or-cleaned. v3 training gate."""
    import json
    import pathlib
    p = pathlib.Path(report_path)
    if not p.exists():
        raise ValueError(f"leakage report missing: {report_path} — run scripts/run_full_leakage_scan.py "
                         "before training v3 candidates")
    report = json.loads(p.read_text(encoding="utf-8"))
    if not leakage_report_is_clean(report):
        raise ValueError(
            f"leakage report {report_path} is NOT clean "
            f"(exact={report.get('exact_hits')}, near={report.get('near_duplicate_hits')}); "
            "re-run with --drop-hits and train on the cleaned candidate file")
