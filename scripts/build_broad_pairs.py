#!/usr/bin/env python3
"""v8 data-scale test: union local (query, positive) pairs into one deduped corpus.

Isolates the DATA-SCALE variable: same model/recipe/hard-negatives as the 22k baseline, only the
pair count grows. Output is candidate format ({query, document, source, domain}) for the leakage
scanner and the v6.1 trainer (which reads `document` as the positive). Stdlib only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SOURCES = [
    ("outputs/autoresearch/prepared/train_candidates.clean.jsonl", "v6_clean"),
    ("data/processed/candidates_v2.jsonl", "candidates_v2"),
    ("data/processed/v6_2/rag_pairs_guardrail_upweighted.jsonl", "v6_2_guardrail"),
    ("data/raw/v3/faq_real_local.jsonl", "faq_real_v3"),
    ("data/raw/v3/ger_backtrans_paraphrase.jsonl", "backtrans_v3"),
    ("data/raw/v2/ger_backtrans_paraphrase.jsonl", "backtrans_v2"),
]


def _s(x):
    if isinstance(x, list):
        x = x[0] if x else ""
    return x.strip() if isinstance(x, str) else ""


def _qp(d):
    q = _s(d.get("query"))
    p = _s(d.get("positive")) or _s(d.get("document"))
    if not p and d.get("candidates") and d.get("positive_doc_ids"):
        pids = d["positive_doc_ids"]
        if isinstance(pids, str):
            try: pids = eval(pids)  # noqa: S307 - local trusted data
            except Exception: pids = []
        cand = d["candidates"]
        if isinstance(cand, str):
            try: cand = eval(cand)  # noqa: S307
            except Exception: cand = []
        for c in (cand or []):
            if isinstance(c, dict) and c.get("doc_id") in pids:
                p = _s(c.get("text")); break
    return q, p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="outputs/v8/data/broad_pairs.jsonl")
    args = ap.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    per = {}
    with out.open("w", encoding="utf-8") as w:
        for src, tag in SOURCES:
            p = Path(src)
            if not p.exists():
                per[tag] = "MISSING"; continue
            n = new = 0
            for ln in p.open(encoding="utf-8"):
                if not ln.strip():
                    continue
                try: d = json.loads(ln)
                except Exception: continue
                q, pos = _qp(d)
                if not q or not pos:
                    continue
                n += 1
                k = (q[:200], pos[:200])
                if k in seen:
                    continue
                seen.add(k); new += 1
                w.write(json.dumps({"query": q, "document": pos, "source": tag,
                                    "domain": _s(d.get("domain")) or tag},
                                   ensure_ascii=False) + "\n")
            per[tag] = f"{new} new / {n} usable"
    print(json.dumps({"output": str(out), "total_unique_pairs": len(seen), "per_source": per},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
