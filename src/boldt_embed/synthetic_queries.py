"""Template-based German synthetic query generation (pure stdlib, deterministic).

Given German passages, generate diverse query→passage candidate pairs across query styles:
factual, keyword, formal-administrative, colloquial, legal/admin, negated, date/number-
sensitive, entity-disambiguation, and FAQ. The result is candidate rows in the standard
schema (see :mod:`data_pipeline`) with ``positive=True`` and full traceability metadata
(``generation_method``, ``template_id``, ``source_passage_id``, ``source_domain``), and the
source passage's license inherited verbatim.

No external APIs, no network, no ML — generation is a deterministic function of the passage.
A later teacher pass (``build_teacher_cache.py``) scores these and low-scoring pairs are
dropped before training. A local-LLM upgrade path is stubbed in
:mod:`boldt_embed.local_llm_generation`.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from .data_pipeline import normalize_text, stable_pair_id, stable_text_hash

SOURCE = "synthetic"

_STOPWORDS = {"Der", "Die", "Das", "Ein", "Eine", "Es", "In", "Im", "Am", "An", "Auf",
              "Und", "Wie", "Was", "Wann", "Wer", "Für", "Mit", "Von", "Zu", "Bei"}
_ENTITIES = {"München", "Berlin", "Hamburg", "Köln", "Bayern", "Sachsen", "Hessen"}


# -------------------------------------------------------------------- slot extraction
def _topic(text: str) -> Optional[str]:
    """Longest mid-sentence capitalized word (German nouns are capitalized) — the passage
    topic. Deterministic: longest, ties broken by first occurrence."""
    words = re.findall(r"\b[A-ZÄÖÜ][\wäöüß]{3,}\b", text)
    candidates = [w for w in words if w not in _STOPWORDS]
    if not candidates:
        return None
    best = candidates[0]
    for w in candidates[1:]:
        if len(w) > len(best):
            best = w
    return best


def _keyword(text: str, topic: Optional[str]) -> Optional[str]:
    for w in re.findall(r"\b[A-ZÄÖÜ][\wäöüß]{3,}\b", text):
        if w not in _STOPWORDS and w != topic:
            return w
    return None


def _number(text: str) -> Optional[str]:
    m = re.search(r"\b\d+\b", text)
    return m.group() if m else None


def _year(text: str) -> Optional[str]:
    m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", text)
    return m.group() if m else None


def _legal_ref(text: str) -> Optional[str]:
    m = re.search(r"§\s*\d+", text)
    return m.group() if m else None


def _entity(text: str) -> Optional[str]:
    for ent in _ENTITIES:
        if ent in text:
            return ent
    return None


# ------------------------------------------------------------------------- templates
# (template_id, generation_method, fn(slots) -> Optional[str])
def _t_factual(s):
    return f"Was versteht man unter {s['topic']}?" if s.get("topic") else None


def _t_keyword(s):
    if s.get("topic") and s.get("keyword"):
        return f"{s['topic']} {s['keyword']}"
    return None


def _t_formal_admin(s):
    return f"Welche Voraussetzungen gelten für {s['topic']}?" if s.get("topic") else None


def _t_colloquial(s):
    return f"Wie funktioniert das mit {s['topic']} eigentlich?" if s.get("topic") else None


def _t_legal_admin(s):
    if s.get("legal_ref"):
        return f"Was regelt {s['legal_ref']}?"
    return f"Welche gesetzliche Regelung gilt für {s['topic']}?" if s.get("topic") else None


def _t_negated(s):
    return f"Wann gilt {s['topic']} nicht?" if s.get("topic") else None


def _t_date_number(s):
    if s.get("year"):
        return f"Was geschah im Jahr {s['year']} im Zusammenhang mit {s.get('topic', 'diesem Thema')}?"
    if s.get("number"):
        return f"Welche Bedeutung hat die Zahl {s['number']} bei {s.get('topic', 'diesem Thema')}?"
    return None


def _t_entity_disambig(s):
    if s.get("entity"):
        return f"Bezieht sich {s['entity']} hier auf die Stadt oder das Bundesland?"
    return None


def _t_faq(s):
    return f"Häufige Frage: Wie kann ich {s['topic']} beantragen?" if s.get("topic") else None


# --- v2 query-family expansion ---
def _t_definition(s):  # germanquad
    return f"Was bedeutet {s['topic']}?" if s.get("topic") else None


def _t_wer(s):  # germanquad
    return f"Wer oder was ist {s['topic']}?" if s.get("topic") else None


def _t_web_fragment(s):  # web
    if s.get("topic") and s.get("keyword"):
        return f"{s['topic']} {s['keyword']} erklärung"
    return None


def _t_web_typo(s):  # web — orthographic variant of the topic (ß/ss, umlaut)
    from .german_adversarial import swap_eszett, swap_umlauts
    t = s.get("topic")
    if not t:
        return None
    v = swap_eszett(t) or swap_umlauts(t)
    return f"{v} info" if v else None


def _t_faq_problem(s):  # faq
    return f"Was tun, wenn {s['topic']} nicht funktioniert?" if s.get("topic") else None


def _t_faq_lost(s):  # faq
    return f"Ich habe {s['topic']} vergessen – was nun?" if s.get("topic") else None


def _t_admin_unterlagen(s):  # admin
    return f"Welche Unterlagen brauche ich für {s['topic']}?" if s.get("topic") else None


def _t_admin_frist(s):  # admin
    return f"Frist für {s['topic']}" if s.get("topic") else None


def _t_admin_antrag_online(s):  # admin
    return f"Antrag {s['topic']} online stellen" if s.get("topic") else None


def _t_admin_legalref_absatz(s):  # admin
    return f"{s['legal_ref']} Absatz Bedeutung" if s.get("legal_ref") else None


def _t_crosslingual_en(s):  # cross_lingual_de_en — English query, German document
    return f"What is {s['topic']}?" if s.get("topic") else None


def _t_negation_distractor(s):  # negation — a DISTRACTOR (positive=False)
    return f"Was hat nichts mit {s['topic']} zu tun?" if s.get("topic") else None


# (template_id, generation_method, family, style, fn). `family` drives --families;
# `style` is kept for the legacy --domains filter. Negation rows are distractors (positive=False)
# and are NOT generated unless the family is explicitly requested.
_TEMPLATES: List = [
    ("factual", "factual_question", "germanquad", "factual", _t_factual),
    ("definition", "definition_question", "germanquad", "factual", _t_definition),
    ("wer", "wer_question", "germanquad", "factual", _t_wer),
    ("date_number", "date_number_query", "germanquad", "date_number", _t_date_number),
    ("entity_disambig", "entity_disambiguation_query", "germanquad", "entity", _t_entity_disambig),
    ("keyword", "keyword_search", "web", "keyword", _t_keyword),
    ("web_fragment", "web_fragment_query", "web", "keyword", _t_web_fragment),
    ("web_typo", "web_typo_query", "web", "keyword", _t_web_typo),
    ("faq", "faq_query", "faq", "faq", _t_faq),
    ("faq_problem", "faq_problem_query", "faq", "faq", _t_faq_problem),
    ("faq_lost", "faq_lost_query", "faq", "faq", _t_faq_lost),
    ("colloquial", "colloquial_query", "faq", "colloquial", _t_colloquial),
    ("formal_admin", "formal_administrative", "admin", "admin", _t_formal_admin),
    ("admin_unterlagen", "admin_unterlagen_query", "admin", "admin", _t_admin_unterlagen),
    ("admin_frist", "admin_frist_query", "admin", "admin", _t_admin_frist),
    ("admin_antrag_online", "admin_antrag_online_query", "admin", "admin", _t_admin_antrag_online),
    ("legal_admin", "legal_admin_query", "admin", "legal", _t_legal_admin),
    ("legal_ref_absatz", "legal_ref_absatz_query", "admin", "legal", _t_admin_legalref_absatz),
    ("crosslingual_en", "crosslingual_en_query", "cross_lingual_de_en", "cross_lingual", _t_crosslingual_en),
    ("negated", "negated_query", "negation", "negation", _t_negated),
    ("negation_distractor", "negation_distractor_query", "negation", "negation", _t_negation_distractor),
]

ALL_QUERY_STYLES = sorted({t[3] for t in _TEMPLATES})
ALL_FAMILIES = sorted({t[2] for t in _TEMPLATES})
# Default families generate POSITIVE pairs only (negation is opt-in -> candidate negatives).
DEFAULT_FAMILIES = [f for f in ALL_FAMILIES if f != "negation"]


def _passage_text(passage: Dict[str, Any]) -> str:
    for key in ("document", "text", "passage", "context"):
        v = passage.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def generate_queries_for_passage(passage: Dict[str, Any],
                                 queries_per_passage: Optional[int] = None,
                                 domains: Optional[Sequence[str]] = None,
                                 families: Optional[Sequence[str]] = None,
                                 min_document_chars: int = 0,
                                 max_document_chars: Optional[int] = None
                                 ) -> List[Dict[str, Any]]:
    """Generate synthetic query→passage candidate rows for one passage. Deterministic.

    ``families`` selects query families (default: all positive families; ``negation`` is opt-in
    and produces distractor rows with ``positive=False``). ``domains`` keeps the legacy
    per-style filter. Char bounds skip passages outside [min, max]."""
    text = normalize_text(_passage_text(passage))
    if not text or len(text) < min_document_chars:
        return []
    if max_document_chars is not None and len(text) > max_document_chars:
        text = text[:max_document_chars]
    fam_filter = set(families) if families is not None else set(DEFAULT_FAMILIES)
    source_domain = str(passage.get("domain") or "unknown")
    license_ = str(passage.get("license") or "unknown")
    passage_id = str(passage.get("doc_id") or passage.get("id") or passage.get("passage_id")
                     or ("d" + stable_text_hash(text)))
    slots = {
        "topic": _topic(text), "keyword": None, "number": _number(text),
        "year": _year(text), "legal_ref": _legal_ref(text), "entity": _entity(text),
    }
    slots["keyword"] = _keyword(text, slots["topic"])

    rows: List[Dict[str, Any]] = []
    for template_id, method, family, style, fn in _TEMPLATES:
        if family not in fam_filter:
            continue
        if domains is not None and style not in domains:
            continue
        query = fn(slots)
        if not query:
            continue
        query = normalize_text(query)
        positive = family != "negation"  # negation family = candidate negatives
        rows.append({
            "query_id": "q" + stable_text_hash(query),
            "doc_id": passage_id,
            "query": query,
            "document": text,
            "positive": positive,
            "source": SOURCE,
            "domain": source_domain,
            "license": license_,
            "pair_hash": stable_pair_id(query, text),
            "metadata": {
                "generation_method": method,
                "template_id": template_id,
                "family": family,
                "source_passage_id": passage_id,
                "source_domain": source_domain,
                "query_style": style,
                "synthetic": True,
            },
        })
    if queries_per_passage is not None and len(rows) > queries_per_passage:
        rows = _round_robin_by_family(rows, queries_per_passage)
    return rows


def _round_robin_by_family(rows: List[Dict[str, Any]], cap: int) -> List[Dict[str, Any]]:
    """Pick up to `cap` rows, cycling through families so a small cap still spans families
    (rather than taking the first N in template order). Deterministic."""
    by_fam: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for r in rows:
        fam = r["metadata"]["family"]
        if fam not in by_fam:
            by_fam[fam] = []
            order.append(fam)
        by_fam[fam].append(r)
    out: List[Dict[str, Any]] = []
    i = 0
    while len(out) < cap and any(by_fam[f] for f in order):
        fam = order[i % len(order)]
        if by_fam[fam]:
            out.append(by_fam[fam].pop(0))
        i += 1
    return out


def generate_synthetic_candidates(passages: Sequence[Dict[str, Any]],
                                  queries_per_passage: Optional[int] = None,
                                  domains: Optional[Sequence[str]] = None,
                                  families: Optional[Sequence[str]] = None,
                                  min_document_chars: int = 0,
                                  max_document_chars: Optional[int] = None
                                  ) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in passages:
        out.extend(generate_queries_for_passage(p, queries_per_passage, domains, families,
                                                min_document_chars, max_document_chars))
    return out
