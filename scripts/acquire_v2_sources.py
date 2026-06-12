#!/usr/bin/env python3
"""Materialize the v2 manifest's TRAINING sources into data/raw/v2/<source_id>.jsonl.

Real data (network, no GPU, no ML):
  * dt_de_dpr               deutsche-telekom/wikipedia-22-12-de-dpr (train): question +
                            formal/informal imperative variants -> real German Wikipedia QA.
  * ger_backtrans_paraphrase deutsche-telekom/ger-backtrans-paraphrase (non-wiki corpora:
                            TED/news/Europarl/...): (de, en_de) paraphrase positives -> web.
  * swim_ir_de              nthakur/swim-ir-monolingual (de): synthetic query -> wiki passage.
  * synthetic_adversarial   copied from data/processed/adversarial_candidates.jsonl.

Honestly-synthetic data (deterministic templates over REAL Wikipedia passages; documents are
real, only query phrasing is generated) for the domains with no licensed corpus on disk:
  * local_admin_v2 (admin), synthetic_faq_v2 (faq), synthetic_legal_v2 (legal_adjacency),
    synthetic_cross_lingual_v2 (cross_lingual_de_en).

Every row carries {source=<manifest source_id>, query, document, domain, license} so the
manifest-gated, leakage-filtered build_v2_candidates.py admits it. The teacher pass later
drops low-scoring synthetic pairs. We OVERPRODUCE each domain so the domain-balanced sampler
can hit its target after dedup + leakage filtering.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import synthetic_queries as sq  # noqa: E402

RAW = ROOT / "data" / "raw" / "v2"
WIKI_LICENSE = "CC-BY-SA-4.0"
LEGAL_MARKERS = ("§", "Gesetz", "Verordnung", "Paragraph", "Absatz", "BGB", "SGB", "StGB",
                 "Verfassung", "Richtlinie", "gesetzlich", "Rechtsverordnung")


def _w(path: pathlib.Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):6d} -> {path.relative_to(ROOT)}")
    return len(rows)


def _row(source, query, document, domain, license_=WIKI_LICENSE):
    return {"source": source, "query": query.strip(), "document": document.strip(),
            "domain": domain, "license": license_}


# --------------------------------------------------------------------- real HF sources
def _first_str(v):
    """These DPR fields are lists (question, imperative_formal/informal). Take the first."""
    if isinstance(v, list):
        v = v[0] if v else ""
    return (v or "").strip() if isinstance(v, str) else ""


def wiki_pool(n_contexts):
    """Real German Wikipedia (context, question, formal/informal imperatives). Cached."""
    from datasets import load_dataset
    ds = load_dataset("deutsche-telekom/wikipedia-22-12-de-dpr")["train"]
    pool = []
    for row in ds:
        ctx = (row.get("context") or "").strip()
        if len(ctx) < 80:
            continue
        pool.append({"context": ctx, "question": _first_str(row.get("question")),
                     "formal": _first_str(row.get("imperative_formal")),
                     "informal": _first_str(row.get("imperative_informal"))})
        if len(pool) >= n_contexts:
            break
    return pool


def dt_rows(pool, n):
    out = []
    for p in pool:
        for q in (p["question"], p["formal"], p["informal"]):
            if q:
                out.append(_row("dt_de_dpr", q, p["context"], "wiki_non_eval"))
        if len(out) >= n:
            break
    return out[:n]


def ger_backtrans_rows(n):
    from datasets import load_dataset
    out = []
    try:
        ds = load_dataset("deutsche-telekom/ger-backtrans-paraphrase", split="train", streaming=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [ger_backtrans] could not load: {exc}"); return out
    for ex in ds:
        de, en_de = (ex.get("de") or "").strip(), (ex.get("en_de") or "").strip()
        if str(ex.get("corpus") or "misc").lower() == "wikipedia":
            continue  # keep strictly non-wiki -> real domain diversity
        if de and en_de and de != en_de and len(de) > 25 and len(en_de) > 25:
            out.append(_row("ger_backtrans_paraphrase", de, en_de, "web"))
        if len(out) >= n:
            break
    return out


def swim_rows(n):
    from datasets import load_dataset
    out = []
    try:
        ds = load_dataset("nthakur/swim-ir-monolingual", "de", split="train", streaming=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [swim_ir] could not load: {exc}"); return out
    for ex in ds:
        q, text = (ex.get("query") or "").strip(), (ex.get("text") or "").strip()
        if q and text and len(text) > 40:
            out.append(_row("swim_ir_de", q, text, "wiki_non_eval"))
        if len(out) >= n:
            break
    return out


# ------------------------------------------------------------------ honestly-synthetic
def _passages(pool, idx_slice, only_legal=False):
    """Real Wikipedia passages as generation seeds (disjoint slices keep families distinct)."""
    out = []
    for p in pool[idx_slice]:
        ctx = p["context"]
        if only_legal and not any(m in ctx for m in LEGAL_MARKERS):
            continue
        out.append({"document": ctx, "domain": "seed", "license": WIKI_LICENSE,
                    "doc_id": "d" + sq.stable_text_hash(ctx)})
    return out


def synth_rows(passages, families, source_id, domain, n, qpp=None):
    cands = sq.generate_synthetic_candidates(passages, queries_per_passage=qpp,
                                             families=families, min_document_chars=80,
                                             max_document_chars=2000)
    out = []
    for c in cands:
        if not c.get("positive", True):
            continue  # never admit negation distractors as positives
        out.append(_row(source_id, c["query"], c["document"], domain))
        if len(out) >= n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wiki-contexts", type=int, default=60000)
    ap.add_argument("--dt", type=int, default=9000)
    ap.add_argument("--ger-backtrans", type=int, default=16000)
    ap.add_argument("--swim", type=int, default=6000)
    ap.add_argument("--admin", type=int, default=10000)
    ap.add_argument("--faq", type=int, default=10000)
    ap.add_argument("--legal", type=int, default=10000)
    ap.add_argument("--cross-lingual", type=int, default=4000)
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    print("=== real Wikipedia passage pool (cached) ===")
    pool = wiki_pool(args.wiki_contexts)
    print(f"  pool: {len(pool)} contexts")

    print("=== real HF training sources ===")
    _w(RAW / "dt_de_dpr.jsonl", dt_rows(pool, args.dt))
    _w(RAW / "ger_backtrans_paraphrase.jsonl", ger_backtrans_rows(args.ger_backtrans))
    _w(RAW / "swim_ir_de.jsonl", swim_rows(args.swim))

    print("=== honestly-synthetic over REAL passages (disjoint slices) ===")
    # disjoint passage slices so the same passage isn't the seed for multiple families
    n = len(pool)
    a, b, c = n // 4, n // 2, 3 * n // 4
    _w(RAW / "local_admin_v2.jsonl",
       synth_rows(_passages(pool, slice(0, a)), ["admin"], "local_admin_v2", "admin", args.admin))
    _w(RAW / "synthetic_faq_v2.jsonl",
       synth_rows(_passages(pool, slice(a, b)), ["faq"], "synthetic_faq_v2", "faq", args.faq))
    _w(RAW / "synthetic_legal_v2.jsonl",
       synth_rows(_passages(pool, slice(b, c), only_legal=True), ["admin"],
                  "synthetic_legal_v2", "legal_adjacency_no_eval_overlap", args.legal))
    _w(RAW / "synthetic_cross_lingual_v2.jsonl",
       synth_rows(_passages(pool, slice(c, n)), ["cross_lingual_de_en"],
                  "synthetic_cross_lingual_v2", "cross_lingual_de_en", args.cross_lingual))

    print("=== german_stress (copy real adversarial) ===")
    adv_src = ROOT / "data" / "processed" / "adversarial_candidates.jsonl"
    adv = []
    if adv_src.exists():
        from boldt_embed import data_pipeline as dp
        for r in dp.stream_jsonl(adv_src):
            q, d = r.get("query"), r.get("document")
            if isinstance(q, str) and q.strip() and isinstance(d, str) and d.strip():
                adv.append(_row("synthetic_adversarial", q, d, "german_stress"))
    _w(RAW / "synthetic_adversarial.jsonl", adv)

    print(f"\n[done] raw v2 sources -> {RAW.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
