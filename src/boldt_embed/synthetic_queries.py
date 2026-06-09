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

from .data_pipeline import normalize_text, stable_text_hash

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


# Each template carries the domain *style* it represents (used by --domains filter).
_TEMPLATES: List = [
    ("factual", "factual_question", "factual", _t_factual),
    ("keyword", "keyword_search", "keyword", _t_keyword),
    ("formal_admin", "formal_administrative", "admin", _t_formal_admin),
    ("colloquial", "colloquial_query", "colloquial", _t_colloquial),
    ("legal_admin", "legal_admin_query", "legal", _t_legal_admin),
    ("negated", "negated_query", "negation", _t_negated),
    ("date_number", "date_number_query", "date_number", _t_date_number),
    ("entity_disambig", "entity_disambiguation_query", "entity", _t_entity_disambig),
    ("faq", "faq_query", "faq", _t_faq),
]

ALL_QUERY_STYLES = [t[2] for t in _TEMPLATES]


def _passage_text(passage: Dict[str, Any]) -> str:
    for key in ("document", "text", "passage", "context"):
        v = passage.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def generate_queries_for_passage(passage: Dict[str, Any],
                                 queries_per_passage: Optional[int] = None,
                                 domains: Optional[Sequence[str]] = None
                                 ) -> List[Dict[str, Any]]:
    """Generate synthetic query→passage candidate rows for one passage. Deterministic."""
    text = normalize_text(_passage_text(passage))
    if not text:
        return []
    source_domain = str(passage.get("domain") or "unknown")
    license_ = str(passage.get("license") or "unknown")
    passage_id = str(passage.get("doc_id") or passage.get("id") or passage.get("passage_id")
                     or ("d" + stable_text_hash(text)))
    slots = {
        "topic": _topic(text),
        "keyword": None,
        "number": _number(text),
        "year": _year(text),
        "legal_ref": _legal_ref(text),
        "entity": _entity(text),
    }
    slots["keyword"] = _keyword(text, slots["topic"])

    rows: List[Dict[str, Any]] = []
    for template_id, method, style, fn in _TEMPLATES:
        if domains is not None and style not in domains:
            continue
        query = fn(slots)
        if not query:
            continue
        query = normalize_text(query)
        rows.append({
            "query_id": "q" + stable_text_hash(query),
            "doc_id": passage_id,
            "query": query,
            "document": text,
            "positive": True,
            "source": SOURCE,
            "domain": source_domain,
            "license": license_,
            "metadata": {
                "generation_method": method,
                "template_id": template_id,
                "source_passage_id": passage_id,
                "source_domain": source_domain,
                "query_style": style,
            },
        })
        if queries_per_passage is not None and len(rows) >= queries_per_passage:
            break
    return rows


def generate_synthetic_candidates(passages: Sequence[Dict[str, Any]],
                                  queries_per_passage: Optional[int] = None,
                                  domains: Optional[Sequence[str]] = None
                                  ) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in passages:
        out.extend(generate_queries_for_passage(p, queries_per_passage, domains))
    return out
