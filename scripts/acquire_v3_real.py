#!/usr/bin/env python3
"""Materialize REAL v3 domain data into data/raw/v3/<source_id>.jsonl (network, no GPU, no ML).

Real, license-verified sources only:
  * faq_real_local        PaDaS-Lab/webfaq (deu)  -> REAL German FAQ question->answer (CC-BY-4.0)
  * ger_backtrans_paraphrase  deutsche-telekom/ger-backtrans-paraphrase (web; CC-BY-SA-4.0)
  * dt_de_dpr             deutsche-telekom/wikipedia-22-12-de-dpr (wiki QA; CC-BY-SA-4.0)
  * synthetic_adversarial copied from data/processed/adversarial_candidates.jsonl (stress, supplemental)

Rows: {source=<source_id>, query, document, domain, license, url?} — what build_v3_candidates admits.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "v3"


def _w(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):6d} -> {path.relative_to(ROOT)}")
    return len(rows)


def faq_rows(n):
    """REAL German FAQ pairs from WebFAQ (CC-BY-4.0)."""
    from datasets import load_dataset
    ds = load_dataset("PaDaS-Lab/webfaq", "deu", split="default", streaming=True)
    out, seen = [], set()
    for ex in ds:
        q, a = (ex.get("question") or "").strip(), (ex.get("answer") or "").strip()
        if q and a and len(a) > 30 and q not in seen:
            seen.add(q)
            out.append({"source": "faq_real_local", "query": q, "document": a,
                        "domain": "faq_real", "license": "CC-BY-4.0", "url": ex.get("url")})
        if len(out) >= n:
            break
    return out


def web_rows(n):
    from datasets import load_dataset
    out = []
    ds = load_dataset("deutsche-telekom/ger-backtrans-paraphrase", split="train", streaming=True)
    for ex in ds:
        de, en_de = (ex.get("de") or "").strip(), (ex.get("en_de") or "").strip()
        if str(ex.get("corpus") or "").lower() == "wikipedia":
            continue
        if de and en_de and de != en_de and len(de) > 25 and len(en_de) > 25:
            out.append({"source": "ger_backtrans_paraphrase", "query": de, "document": en_de,
                        "domain": "web", "license": "CC-BY-SA-4.0"})
        if len(out) >= n:
            break
    return out


def wiki_rows(n):
    from datasets import load_dataset
    ds = load_dataset("deutsche-telekom/wikipedia-22-12-de-dpr")["train"]
    out = []
    for row in ds:
        ctx = (row.get("context") or "").strip()
        qs = row.get("question") or []
        q = (qs[0] if isinstance(qs, list) and qs else qs if isinstance(qs, str) else "").strip()
        if q and len(ctx) > 80:
            out.append({"source": "dt_de_dpr", "query": q, "document": ctx,
                        "domain": "wiki_non_eval", "license": "CC-BY-SA-4.0"})
        if len(out) >= n:
            break
    return out


def stress_rows():
    src = ROOT / "data" / "processed" / "adversarial_candidates.jsonl"
    out = []
    if src.exists():
        for line in src.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                q, d = r.get("query"), r.get("document")
                if isinstance(q, str) and q.strip() and isinstance(d, str) and d.strip():
                    out.append({"source": "synthetic_adversarial", "query": q, "document": d,
                                "domain": "german_stress", "license": "synthetic-inherits-source"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--faq", type=int, default=15000)
    ap.add_argument("--web", type=int, default=15000)
    ap.add_argument("--wiki", type=int, default=12000)
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    print("=== REAL v3 domain data ===")
    _w(RAW / "faq_real_local.jsonl", faq_rows(args.faq))
    _w(RAW / "ger_backtrans_paraphrase.jsonl", web_rows(args.web))
    _w(RAW / "dt_de_dpr.jsonl", wiki_rows(args.wiki))
    _w(RAW / "synthetic_adversarial.jsonl", stress_rows())
    print(f"[done] -> {RAW.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
