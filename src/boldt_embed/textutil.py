"""German-aware text normalization and tokenization (pure stdlib).

Shared by data leakage checks, the local benchmark, and the eval harness so that
every component tokenizes identically. Umlaut/ß folding makes lexical matching and
leakage detection robust to common German spelling variants.
"""
from __future__ import annotations

import re
from typing import List

_TOKEN_RE = re.compile(r"[\w§]+", re.UNICODE)


def normalize(text: str) -> str:
    return (
        text.lower()
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(normalize(text))


def jaccard(a, b) -> float:
    """Token-set Jaccard similarity. Accepts any iterables; empty -> 0.0."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0
