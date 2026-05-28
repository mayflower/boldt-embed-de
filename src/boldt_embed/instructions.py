"""Instruction / template formatting for queries and documents (pure stdlib).

Uses literal substring replacement (not str.format) so templates that contain other
braces or German punctuation never raise.
"""
from __future__ import annotations


def format_query(template: str, query: str) -> str:
    if not template:
        return query
    if "{query}" in template:
        return template.replace("{query}", query)
    return f"{template}{query}"


def format_document(template: str, document: str) -> str:
    if not template:
        return document
    if "{document}" in template:
        return template.replace("{document}", document)
    return f"{template}{document}"
