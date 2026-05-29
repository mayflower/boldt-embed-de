"""Deterministic German hard-negative generators and synthetic-pair quality filters.

Pure stdlib. Each generator takes a positive German text and returns a *plausible but
wrong* variant for the named family, or ``None`` if the rule does not apply. Generators
are deterministic (rule-based, no RNG) so synthetic data is reproducible.

Families (ADR-004 / DATA_PLAN): compound, negation, legal_ref, dates_numbers,
regional_variant, entity_disambiguation.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .textutil import jaccard, normalize, tokenize


def _changed(original: str, candidate: Optional[str]) -> Optional[str]:
    if candidate is None:
        return None
    return candidate if normalize(candidate) != normalize(original) else None


# --- compound: swap a compound noun for a different, similar-looking compound ---
_COMPOUND_SWAPS: List[Tuple[str, str]] = [
    ("Kündigungsfrist", "Kündigungsschutzklage"),
    ("Mietkaution", "Maklerprovision"),
    ("Krankmeldung", "Urlaubsantrag"),
    ("Umsatzsteuer", "Einkommensteuer"),
    ("Widerrufsrecht", "Gewährleistungsrecht"),
    ("Nettokaltmiete", "Betriebskostenabrechnung"),
]


def neg_compound(text: str) -> Optional[str]:
    for src, dst in _COMPOUND_SWAPS:
        if src.lower() in text.lower():
            return _changed(text, re.sub(re.escape(src), dst, text, count=1, flags=re.IGNORECASE))
    return None


# --- negation: flip the polarity of the statement ---
_NEG_TOGGLES: List[Tuple[str, str]] = [
    ("besteht ein", "besteht kein"),
    ("besteht eine", "besteht keine"),
    (" ist ", " ist nicht "),
    (" sind ", " sind nicht "),
    (" muss ", " muss nicht "),
    (" darf ", " darf nicht "),
    (" kann ", " kann nicht "),
    (" wird ", " wird nicht "),
]


def neg_negation(text: str) -> Optional[str]:
    for src, dst in _NEG_TOGGLES:
        if src in text.lower():
            out = re.sub(re.escape(src), dst, text, count=1, flags=re.IGNORECASE)
            changed = _changed(text, out)
            if changed:
                return changed
    # fallback: explicit clausal negation
    return _changed(text, f"Es trifft nicht zu, dass {text[0].lower()}{text[1:]}")


# --- legal_ref: change the statute reference ---
def neg_legal_ref(text: str) -> Optional[str]:
    def bump(m: re.Match) -> str:
        return f"§ {int(m.group(1)) + 30}"

    out = re.sub(r"§\s*(\d+)", bump, text, count=1)
    if normalize(out) != normalize(text):
        return out
    # also try Absatz / Nr.
    out = re.sub(r"Abs\.\s*(\d+)", lambda m: f"Abs. {int(m.group(1)) + 1}", text, count=1)
    return _changed(text, out)


# --- dates_numbers: perturb a number and/or swap a time unit ---
_UNIT_SWAPS = [("Tage", "Monate"), ("Tagen", "Monaten"), ("Monate", "Wochen"),
               ("Monaten", "Wochen"), ("Wochen", "Tagen")]


def neg_dates_numbers(text: str) -> Optional[str]:
    out = re.sub(r"\b(\d+)\b", lambda m: str(int(m.group(1)) + 16), text, count=1)
    for src, dst in _UNIT_SWAPS:
        if re.search(rf"\b{src}\b", out):
            out = re.sub(rf"\b{src}\b", dst, out, count=1)
            break
    return _changed(text, out)


# --- regional_variant: swap a regional/temporal term for a different concept ---
_REGIONAL_SWAPS: List[Tuple[str, str]] = [
    ("Jänner", "März"),
    ("Januar", "März"),
    ("Feber", "April"),
    ("Februar", "April"),
]


def neg_regional_variant(text: str) -> Optional[str]:
    for src, dst in _REGIONAL_SWAPS:
        if src.lower() in text.lower():
            return _changed(text, re.sub(re.escape(src), dst, text, count=1, flags=re.IGNORECASE))
    return None


# --- entity_disambiguation: send an ambiguous entity to the wrong sense ---
_ENTITY_RULES: List[Tuple[frozenset, str]] = [
    (frozenset({"golf", "volkswagen"}),
     "Golf bezeichnet hier die Präzisionssportart mit Schläger und Ball, nicht das Automodell."),
    (frozenset({"golf", "auto"}),
     "Golf bezeichnet hier die Präzisionssportart mit Schläger und Ball, nicht das Automodell."),
    (frozenset({"bank", "konto"}),
     "Die Bank bezeichnet hier eine Sitzbank im Park, nicht das Geldinstitut."),
    (frozenset({"schloss"}),
     "Das Schloss bezeichnet hier das Türschloss an einer Tür, nicht das herrschaftliche Gebäude."),
]


def neg_entity_disambiguation(text: str) -> Optional[str]:
    toks = set(tokenize(text))
    for triggers, replacement in _ENTITY_RULES:
        if triggers.issubset(toks):
            return _changed(text, replacement)
    return None


# --- outcome_flip: keep the topic but flip the outcome (similar facts, different result) ---
_OUTCOME_SWAPS: List[Tuple[str, str]] = [
    ("ausgeschlossen", "möglich"),
    ("möglich", "ausgeschlossen"),
    ("unzulässig", "zulässig"),
    ("zulässig", "unzulässig"),
    ("unwirksam", "wirksam"),
    ("wirksam", "unwirksam"),
    ("wird gewährt", "wird abgelehnt"),
    ("wird abgelehnt", "wird gewährt"),
    ("erlaubt", "verboten"),
    ("verboten", "erlaubt"),
]


def neg_outcome_flip(text: str) -> Optional[str]:
    """Same facts/topic, opposite outcome (a 'similar but different result' negative)."""
    for src, dst in _OUTCOME_SWAPS:
        if re.search(rf"\b{re.escape(src)}\b", text, flags=re.IGNORECASE):
            return _changed(text, re.sub(rf"\b{re.escape(src)}\b", dst, text, count=1, flags=re.IGNORECASE))
    return None


GENERATORS: Dict[str, Callable[[str], Optional[str]]] = {
    "compound": neg_compound,
    "negation": neg_negation,
    "legal_ref": neg_legal_ref,
    "dates_numbers": neg_dates_numbers,
    "regional_variant": neg_regional_variant,
    "entity_disambiguation": neg_entity_disambiguation,
    "outcome_flip": neg_outcome_flip,
}

# The ten query types a German retrieval set should cover (prompt 05). LLM templates per
# type live in data/synthetic/prompt_specs.json -> query_type_templates.
QUERY_TYPES: Tuple[str, ...] = (
    "keyword",
    "question",
    "short_vague",
    "long_detailed",
    "entity",
    "date_number",
    "legal_admin",
    "support",
    "summary",
    "negation_contradiction",
)


def make_hard_negatives(text: str, categories: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """Return {category: negative} for every requested family that produces a change."""
    cats = list(categories) if categories else list(GENERATORS)
    out: Dict[str, str] = {}
    for cat in cats:
        gen = GENERATORS.get(cat)
        if gen is None:
            continue
        neg = gen(text)
        if neg is not None:
            out[cat] = neg
    return out


# --- synthetic-pair quality filters ---
_GERMAN_MARKERS = {
    "der", "das", "und", "nicht", "ein", "eine", "ist", "mit", "fuer", "von",
    "auf", "dem", "den", "des", "wird", "betraegt", "oder", "auch", "kein",
    "keine", "muss", "darf", "bei", "fuers", "einen", "einer",
}


def looks_german(text: str) -> bool:
    if any(ch in text for ch in "äöüßÄÖÜ"):
        return True
    return bool(_GERMAN_MARKERS.intersection(tokenize(text)))


def filter_pair(
    query: str,
    positive: str,
    negatives: Optional[Sequence[str]] = None,
    *,
    min_chars: int = 8,
    max_chars: int = 2000,
    max_neg_positive_jaccard: float = 0.97,
) -> Tuple[bool, List[str]]:
    """Quality gate for a synthetic pair. Returns (ok, reasons-it-failed)."""
    reasons: List[str] = []
    if not (min_chars <= len(query) <= max_chars):
        reasons.append("query_length")
    if not (min_chars <= len(positive) <= max_chars):
        reasons.append("positive_length")
    if normalize(query) == normalize(positive):
        reasons.append("query_equals_positive")
    if not looks_german(f"{query} {positive}"):
        reasons.append("not_german")
    pos_sig = set(tokenize(positive))
    for neg in negatives or []:
        if normalize(neg) == normalize(positive):
            reasons.append("negative_equals_positive")
        elif jaccard(pos_sig, set(tokenize(neg))) >= max_neg_positive_jaccard:
            reasons.append("negative_too_similar")
    return (not reasons, reasons)
