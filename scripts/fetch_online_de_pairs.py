#!/usr/bin/env python3
"""v8 — actually FETCH real online German retrieval training pairs (not recombine local data).

Sources (training-appropriate, NOT the GermanQuAD/GerDaLIR/MIRACL benchmark eval splits):
  - SWIM-IR German (nthakur/swim-ir-monolingual, 'de'): synthetic Wikipedia ad-hoc retrieval —
    the MIRACL distribution. The local copy was a 6k sample; this pulls the full set.
  - GermanDPR (deepset/germandpr): Wikipedia QA (question -> positive passage).

Writes candidate format {query, document, source, domain} for the leakage scanner + trainer.
Needs the [eval]/[train] extra (datasets). Streams SWIM-IR so we can cap without a full download."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _w(fh, q, d, source, domain, seen):
    q = (q or "").strip()
    d = (d or "").strip()
    if not q or len(d) < 5:
        return 0
    k = (q[:200], d[:200])
    if k in seen:
        return 0
    seen.add(k)
    fh.write(json.dumps({"query": q, "document": d, "source": source, "domain": domain},
                        ensure_ascii=False) + "\n")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="outputs/v8/data/online_pairs.jsonl")
    ap.add_argument("--swim-max", type=int, default=400000)
    ap.add_argument("--no-germandpr", action="store_true")
    args = ap.parse_args()
    from datasets import load_dataset

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    counts = {}
    with out.open("w", encoding="utf-8") as fh:
        # --- SWIM-IR German (streamed, capped) ---
        n = 0
        try:
            ds = load_dataset("nthakur/swim-ir-monolingual", "de", split="train", streaming=True)
            for ex in ds:
                n += _w(fh, ex.get("query"), ex.get("text"), "swim_ir_de_full", "wiki_non_eval", seen)
                if n >= args.swim_max:
                    break
            counts["swim_ir_de_full"] = n
        except Exception as e:
            counts["swim_ir_de_full"] = f"ERR {type(e).__name__}: {e}"

        # --- GermanDPR (Wikipedia QA, train) ---
        if not args.no_germandpr:
            g = 0
            try:
                gd = load_dataset("deepset/germandpr", split="train")
                for ex in gd:
                    pos = ex.get("positive_ctxs") or {}
                    texts = pos.get("text") if isinstance(pos, dict) else None
                    if not texts and isinstance(pos, list):
                        texts = [c.get("text") for c in pos if isinstance(c, dict)]
                    if isinstance(texts, str):
                        texts = [texts]
                    for t in (texts or [])[:1]:
                        g += _w(fh, ex.get("question"), t, "germandpr", "wiki_non_eval", seen)
                counts["germandpr"] = g
            except Exception as e:
                counts["germandpr"] = f"ERR {type(e).__name__}: {e}"

    print(json.dumps({"output": str(out), "total_unique": len(seen), "per_source": counts},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
