"""v5 teacher-validated German RAG question generation (pure stdlib core, no API calls).

v2 proved that *template-only* synthetic queries over Wikipedia mostly fail teacher validation;
v3 proved real FAQ works. v5 needs non-FAQ questions, but a generated question is **provisional**:
it is never training data until a Qwen3-Reranker teacher score passes threshold. This module
builds the prompts and shapes/validates generated rows; it does **not** call any external API and
imports no ML by default.

Generation modes:

- ``dry_run_templates``      — deterministic simple templates, for tests/wiring only (these are
                               intentionally weak — the v2 lesson — and must still be teacher-validated);
- ``local_llm_jsonl``        — consume pre-generated local-LLM outputs from JSONL (join provenance
                               from our trusted passage records, not from the LLM);
- ``teacher_prompt_export``  — write German JSON-output prompts for an external/local LLM; no calls;
- ``optional_local_transformers`` — only behind an explicit ``--allow-local-llm`` flag (lazy import).

Every generated row carries ``synthetic_query=true`` and ``must_teacher_validate=true``. License
and provenance come from the trusted passage record, never from the LLM.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple

from .v5_data_mixer import is_unknown_license, leakage_reason

QUERY_STYLES: Tuple[str, ...] = (
    "germanquad_fact",          # GermanQuAD-style fact question
    "definition",               # definition question
    "how_to",                   # how-to / procedure question
    "comparison",               # comparison question
    "reason_why",               # reason / why question
    "evidence_support",         # evidence / support question
    "long_doc_locating",        # long-doc locating question
    "ambiguous_needs_context",  # ambiguous query requiring title/context
    "short_web_search",         # short web-search query
    "german_stress",            # negation / date / compound / entity
)

GENERATION_METHODS: Tuple[str, ...] = (
    "dry_run_templates", "local_llm_jsonl", "teacher_prompt_export", "optional_local_transformers",
)

# German, per-style instruction lines shared by the prompt builder and the dry-run templates.
STYLE_INSTRUCTIONS: Dict[str, str] = {
    "germanquad_fact": "Stelle eine konkrete Faktenfrage zu einer im Abschnitt genannten Tatsache.",
    "definition": "Frage nach der Bedeutung oder Definition eines im Abschnitt erklärten Begriffs.",
    "how_to": "Frage nach dem Vorgehen oder den Schritten, die der Abschnitt beschreibt.",
    "comparison": "Frage nach einem Unterschied oder Vergleich, der aus dem Abschnitt hervorgeht.",
    "reason_why": "Frage nach dem Grund oder der Ursache für eine im Abschnitt genannte Aussage.",
    "evidence_support": "Frage danach, welche Belege der Abschnitt für eine Aussage liefert.",
    "long_doc_locating": "Stelle eine Frage, deren Antwort nur in einem bestimmten Teil des "
                         "Abschnitts steht (Lokalisierungsfrage).",
    "ambiguous_needs_context": "Stelle eine kurze, mehrdeutige Frage, die ohne Titel/Kontext "
                               "nicht eindeutig ist, aber mit dem Abschnitt beantwortbar bleibt.",
    "short_web_search": "Formuliere eine sehr kurze Web-Suchanfrage (Stichworte) zum Abschnitt.",
    "german_stress": "Formuliere eine Frage mit deutscher Härte: Negation, Datum, Kompositum "
                     "oder Entitäts-Disambiguierung.",
}


def _first_words(text: str, n: int) -> str:
    return " ".join((text or "").split()[:n])


def _stable_hash(*parts: str) -> str:
    return hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=8).hexdigest()


def validate_passage(p: Any, idx: int) -> List[str]:
    """Schema + leakage check for an input passage record."""
    errs: List[str] = []
    if not isinstance(p, dict):
        return [f"passage[{idx}]: not a JSON object"]
    for k in ("source_passage_id", "document", "domain", "license"):
        if not isinstance(p.get(k), str) or not p[k].strip():
            errs.append(f"passage[{idx}] ({p.get('source_passage_id', '?')}): '{k}' must be a non-empty string")
    if is_unknown_license(p.get("license")):
        errs.append(f"passage[{idx}] ({p.get('source_passage_id', '?')}): unknown/unpermitted license")
    reason = leakage_reason(p)
    if reason:
        errs.append(f"passage[{idx}] ({p.get('source_passage_id', '?')}): public-benchmark/eval "
                    f"leakage ({reason}) — never generate questions from eval text")
    return errs


def build_prompt(passage: Dict[str, Any], style: str) -> str:
    """Deterministic German prompt that asks for ONE passage-answerable question + JSON output."""
    if style not in QUERY_STYLES:
        raise ValueError(f"unknown query_style: {style}")
    title = passage.get("title") or "(kein Titel)"
    document = passage.get("document", "")
    # The required JSON schema is a literal block (single braces) so no API/templating is needed.
    json_spec = (
        '{"query": "<deine Frage auf Deutsch>", '
        f'"query_style": "{style}", '
        '"answerable_only_from_passage": true, '
        '"answerable_without_passage": false}'
    )
    return (
        "Du bist Experte für deutsche Informationssuche und RAG-Qualität.\n"
        "Lies den folgenden Textabschnitt und formuliere GENAU EINE deutschsprachige Frage, die "
        "AUSSCHLIESSLICH mit diesem Abschnitt beantwortet werden kann. Die Frage darf NICHT aus "
        "Allgemeinwissen beantwortbar sein und muss den Abschnitt zwingend benötigen.\n"
        f"Frage-Stil: {style} — {STYLE_INSTRUCTIONS[style]}\n\n"
        f"Titel: {title}\n"
        "Textabschnitt:\n\"\"\"\n"
        f"{document}\n\"\"\"\n\n"
        "Antworte AUSSCHLIESSLICH mit einem einzigen JSON-Objekt in genau diesem Format "
        "(keine Erklärung, kein Markdown):\n"
        f"{json_spec}\n"
        "Setze \"answerable_without_passage\" auf true, falls die Frage auch ohne den Abschnitt "
        "beantwortbar wäre — solche Fragen werden verworfen."
    )


def export_prompts(passages: List[Dict[str, Any]], styles: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Deterministic prompt rows for an external/local LLM (no calls)."""
    out: List[Dict[str, Any]] = []
    for p in passages:
        for style in styles:
            out.append({
                "prompt_id": _stable_hash(p["source_passage_id"], style),
                "source_passage_id": p["source_passage_id"],
                "domain": p.get("domain"), "query_style": style,
                "expects_json": True, "prompt": build_prompt(p, style),
            })
    return out


def _template_query(passage: Dict[str, Any], style: str) -> str:
    """Intentionally weak deterministic template (dry-run/tests only)."""
    title = passage.get("title") or _first_words(passage.get("document", ""), 4) or "dieses Thema"
    doc4 = _first_words(passage.get("document", ""), 4) or title
    return {
        "germanquad_fact": f"Welche Tatsache nennt der Abschnitt zu {title}?",
        "definition": f"Was bedeutet {title} laut dem Abschnitt?",
        "how_to": f"Wie geht man laut dem Abschnitt bei {title} vor?",
        "comparison": f"Worin unterscheidet sich {title} von Verwandtem im Abschnitt?",
        "reason_why": f"Warum gilt die Aussage zu {title} im Abschnitt?",
        "evidence_support": f"Welche Belege nennt der Abschnitt für {title}?",
        "long_doc_locating": f"An welcher Stelle behandelt der Abschnitt {title}?",
        "ambiguous_needs_context": f"Worum geht es bei {title}?",
        "short_web_search": doc4,
        "german_stress": f"Welche Frist gilt NICHT für {title} laut Abschnitt?",
    }[style]


def make_row(passage: Dict[str, Any], query: str, style: str, generation_method: str,
             answerable_without_passage: Any = None) -> Dict[str, Any]:
    """Build a provisional generated-question row (provenance/license from the passage)."""
    row: Dict[str, Any] = {
        "source_passage_id": passage["source_passage_id"],
        "query": query,
        "document": passage["document"],
        "query_style": style,
        "generation_method": generation_method,
        "synthetic_query": True,
        "license": passage["license"],
        "source_id": passage.get("source_id") or passage["source_passage_id"],
        "domain": passage["domain"],
        "must_teacher_validate": True,
    }
    if passage.get("title"):
        row["title"] = passage["title"]
    if answerable_without_passage is not None:
        row["answerable_without_passage"] = bool(answerable_without_passage)
    return row


def generate_from_templates(passages: List[Dict[str, Any]],
                            styles: Tuple[str, ...]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in passages:
        for style in styles:
            rows.append(make_row(p, _template_query(p, style), style, "dry_run_templates"))
    return rows


def rows_from_local_llm(llm_rows: List[Dict[str, Any]], passages_by_id: Dict[str, Dict[str, Any]]
                        ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Join LLM outputs to trusted passages; reject answerable-without-passage; return errors."""
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    errors: List[str] = []
    for i, r in enumerate(llm_rows):
        if not isinstance(r, dict):
            errors.append(f"llm_row[{i}]: not a JSON object"); continue
        pid = r.get("source_passage_id")
        passage = passages_by_id.get(str(pid))
        if passage is None:
            errors.append(f"llm_row[{i}]: source_passage_id '{pid}' not found in passages"); continue
        query = r.get("query")
        if not isinstance(query, str) or not query.strip():
            errors.append(f"llm_row[{i}] ({pid}): empty query"); continue
        style = r.get("query_style")
        if style not in QUERY_STYLES:
            errors.append(f"llm_row[{i}] ({pid}): unknown query_style '{style}'"); continue
        awp = r.get("answerable_without_passage")
        method = r.get("generation_method") or r.get("model") or "local_llm_jsonl"
        row = make_row(passage, query.strip(), style, str(method), answerable_without_passage=awp)
        if awp is True:
            rejected.append(row)            # req #5: not a valid RAG question
        else:
            kept.append(row)
    return kept, rejected, errors


def validate_generated_row(row: Any, idx: int) -> List[str]:
    """Schema/contract errors for a generated-question row."""
    errs: List[str] = []
    if not isinstance(row, dict):
        return [f"row[{idx}]: not a JSON object"]
    for k in ("source_passage_id", "query", "document", "query_style", "generation_method",
              "license", "source_id", "domain"):
        if not isinstance(row.get(k), str) or not row[k].strip():
            errs.append(f"row[{idx}]: '{k}' must be a non-empty string")
    if row.get("synthetic_query") is not True:
        errs.append(f"row[{idx}]: 'synthetic_query' must be true")
    if row.get("must_teacher_validate") is not True:
        errs.append(f"row[{idx}]: 'must_teacher_validate' must be true (provisional until teacher-scored)")
    if row.get("query_style") not in QUERY_STYLES:
        errs.append(f"row[{idx}]: unknown query_style '{row.get('query_style')}'")
    # generation_method is a free-form provenance string (a pipeline mode OR a model id); it only
    # has to be a non-empty string (checked above) — not restricted to GENERATION_METHODS.
    if is_unknown_license(row.get("license")):
        errs.append(f"row[{idx}]: unknown/unpermitted license '{row.get('license')}'")
    if "answerable_without_passage" in row and not isinstance(row["answerable_without_passage"], bool):
        errs.append(f"row[{idx}]: 'answerable_without_passage' must be a bool if present")
    return errs


def is_training_ready(row: Dict[str, Any], teacher_threshold: float) -> bool:
    """A provisional row becomes training-ready ONLY when a teacher score passes threshold."""
    if row.get("must_teacher_validate"):
        ts = row.get("teacher_score")
        return isinstance(ts, (int, float)) and not isinstance(ts, bool) and ts >= teacher_threshold
    return True


def summarize(rows: List[Dict[str, Any]], *, mode: str, rejected: int = 0,
              errors: List[str] | None = None) -> Dict[str, Any]:
    def _counts(key):
        out: Dict[str, int] = {}
        for r in rows:
            out[str(r.get(key))] = out.get(str(r.get(key)), 0) + 1
        return dict(sorted(out.items()))

    examples: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        b = examples.setdefault(str(r.get("query_style")), [])
        if len(b) < 2:
            b.append({"source_passage_id": str(r.get("source_passage_id")), "query": r.get("query", "")})
    styles_present = sorted({str(r.get("query_style")) for r in rows})
    return {
        "mode": mode,
        "generated_rows": len(rows),
        "rejected_answerable_without_passage": rejected,
        "training_ready_rows": 0,            # all generated rows are provisional by construction
        "all_must_teacher_validate": all(r.get("must_teacher_validate") is True for r in rows),
        "rows_by_query_style": _counts("query_style"),
        "rows_by_generation_method": _counts("generation_method"),
        "rows_by_domain": _counts("domain"),
        "rows_by_license": _counts("license"),
        "query_styles_present": styles_present,
        "query_styles_missing": [s for s in QUERY_STYLES if s not in styles_present],
        "examples_per_style": examples,
        "errors": errors or [],
    }
