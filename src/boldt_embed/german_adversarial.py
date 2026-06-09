"""Deterministic German adversarial candidate generator (pure stdlib).

Produces query/document variants that stress the specifically-German failure modes a
German retriever must get right: ß/ss, umlaut spellings, compounds, negation, dates &
numbers, legal references (§ / Absatz / Satz / SGB / BGB), formal vs informal register, and
entity disambiguation.

Two kinds of variant:

* **paraphrase** — meaning-preserving (orthography/register/legal-ref wording). The variant
  query should still retrieve the original document → ``positive=True``.
* **distractor** — meaning-changing (negation, altered number/date, swapped entity). The
  variant looks lexically close but no longer matches → ``positive=False`` (a hard negative).

All output is marked ``source="synthetic_adversarial"``, ``domain="german_stress"``, and
carries ``metadata.template_id`` / ``generation_method`` / ``source_passage_id`` for
traceability. Generation is a pure function of the input — no randomness — so it is
reproducible and testable.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from .data_pipeline import stable_text_hash

SOURCE = "synthetic_adversarial"
DOMAIN = "german_stress"

_UMLAUT_MAP = {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"}
_INFORMAL_TO_FORMAL = {"du": "Sie", "Du": "Sie", "dein": "Ihr", "Dein": "Ihr",
                       "dich": "Sie", "dir": "Ihnen"}
_FORMAL_TO_INFORMAL = {"Sie": "du", "Ihnen": "dir", "Ihr": "dein"}
# Small deterministic entity map for disambiguation distractors (regional/geographic).
_ENTITY_SWAP = {"München": "Hamburg", "Berlin": "Köln", "Bayern": "Sachsen",
                "Hamburg": "München", "Köln": "Berlin"}
_NEGATABLE = ["ist", "sind", "darf", "kann", "muss", "wird", "hat", "war"]


# ------------------------------------------------------------- transform primitives
def swap_eszett(text: str) -> Optional[str]:
    if "ß" in text:
        return text.replace("ß", "ss")
    if "ss" in text:
        return text.replace("ss", "ß", 1)
    return None


def swap_umlauts(text: str) -> Optional[str]:
    if not any(u in text for u in _UMLAUT_MAP):
        return None
    out = text
    for u, rep in _UMLAUT_MAP.items():
        out = out.replace(u, rep)
    return out


def split_compound(text: str) -> Optional[str]:
    """Insert a space inside the longest long word (>=12 chars) at its midpoint — a crude
    but deterministic stand-in for compound splitting (Mietkaution → Miet kaution)."""
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    long = [w for w in words if len(w) >= 12]
    if not long:
        return None
    target = max(long, key=len)
    mid = len(target) // 2
    split = target[:mid] + " " + target[mid:]
    return text.replace(target, split, 1)


def change_register(text: str) -> Optional[str]:
    toks = text.split()
    if any(t.strip(",.?!") in _FORMAL_TO_INFORMAL for t in toks):
        mapping = _FORMAL_TO_INFORMAL
    elif any(t.strip(",.?!") in _INFORMAL_TO_FORMAL for t in toks):
        mapping = _INFORMAL_TO_FORMAL
    else:
        return None
    out = [mapping.get(t.strip(",.?!"), t) for t in toks]
    return " ".join(out) if out != toks else None


def legal_ref_paraphrase(text: str) -> Optional[str]:
    """'§ 551' → 'Paragraph 551' (meaning-preserving wording variant)."""
    if "§" not in text:
        return None
    return re.sub(r"§\s*(\d+)", r"Paragraph \1", text)


def add_negation(text: str) -> Optional[str]:
    """Insert 'nicht' after the first negatable verb → meaning flips (distractor)."""
    toks = text.split()
    for i, t in enumerate(toks):
        if t.strip(",.?!").lower() in _NEGATABLE:
            toks.insert(i + 1, "nicht")
            return " ".join(toks)
    return None


def change_number(text: str) -> Optional[str]:
    """Increment the first integer → wrong quantity (distractor)."""
    m = re.search(r"\d+", text)
    if not m:
        return None
    val = str(int(m.group()) + 1)
    return text[:m.start()] + val + text[m.end():]


def change_date_year(text: str) -> Optional[str]:
    """Shift the first 4-digit year by one → wrong date (distractor)."""
    m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", text)
    if not m:
        return None
    val = str(int(m.group()) + 1)
    return text[:m.start()] + val + text[m.end():]


def change_legal_ref(text: str) -> Optional[str]:
    """'§ 551' → '§ 552' (wrong section → distractor)."""
    m = re.search(r"§\s*(\d+)", text)
    if not m:
        return None
    val = str(int(m.group(1)) + 1)
    return text[:m.start()] + "§ " + val + text[m.end():]


def swap_entity(text: str) -> Optional[str]:
    for ent, rep in _ENTITY_SWAP.items():
        if ent in text:
            return text.replace(ent, rep, 1)
    return None


# (template_id, generation_method, target, kind, fn)
# target: which side to transform; kind: paraphrase(positive) / distractor(negative)
_TRANSFORMS = [
    ("ss_eszett", "orthographic_eszett", "query", "paraphrase", swap_eszett),
    ("umlaut_ascii", "orthographic_umlaut", "query", "paraphrase", swap_umlauts),
    ("compound_split", "compound_split", "query", "paraphrase", split_compound),
    ("register_shift", "register_formal_informal", "query", "paraphrase", change_register),
    ("legal_ref_wording", "legal_reference_paraphrase", "query", "paraphrase", legal_ref_paraphrase),
    ("negation", "negation", "document", "distractor", add_negation),
    ("number_shift", "number_change", "document", "distractor", change_number),
    ("date_shift", "date_change", "document", "distractor", change_date_year),
    ("legal_ref_wrong", "legal_reference_wrong_section", "document", "distractor", change_legal_ref),
    ("entity_swap", "entity_disambiguation", "document", "distractor", swap_entity),
]


def _candidate(query: str, document: str, positive: bool, template_id: str,
               method: str, source_passage_id: str, license_: str) -> Dict[str, Any]:
    return {
        "query_id": "q" + stable_text_hash(query),
        "doc_id": "d" + stable_text_hash(document),
        "query": query,
        "document": document,
        "positive": positive,
        "source": SOURCE,
        "domain": DOMAIN,
        "license": license_,
        "metadata": {
            "generation_method": method,
            "template_id": template_id,
            "source_passage_id": source_passage_id,
        },
    }


def generate_for_seed(seed: Dict[str, Any], include: Optional[Sequence[str]] = None,
                      emit_anchor: bool = True) -> List[Dict[str, Any]]:
    """Generate adversarial candidates for one seed ``{"query","document",...}``.

    ``include`` optionally restricts to a set of template_ids. ``emit_anchor`` emits the
    original (query, document, positive) pair so the true match is always present.
    """
    query = str(seed.get("query", "")).strip()
    document = str(seed.get("document", "")).strip()
    if not query or not document:
        return []
    license_ = str(seed.get("license") or "unknown")
    passage_id = str(seed.get("doc_id") or seed.get("id") or ("d" + stable_text_hash(document)))
    rows: List[Dict[str, Any]] = []
    if emit_anchor:
        rows.append(_candidate(query, document, True, "anchor", "anchor", passage_id, license_))
    for template_id, method, target, kind, fn in _TRANSFORMS:
        if include is not None and template_id not in include:
            continue
        base = query if target == "query" else document
        variant = fn(base)
        if not variant or variant == base:
            continue
        positive = kind == "paraphrase"
        if target == "query":
            rows.append(_candidate(variant, document, positive, template_id, method,
                                   passage_id, license_))
        else:
            rows.append(_candidate(query, variant, positive, template_id, method,
                                   passage_id, license_))
    return rows


def generate_adversarial_candidates(seeds: Sequence[Dict[str, Any]],
                                    include: Optional[Sequence[str]] = None,
                                    emit_anchor: bool = True) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seed in seeds:
        out.extend(generate_for_seed(seed, include=include, emit_anchor=emit_anchor))
    return out
