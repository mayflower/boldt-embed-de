#!/usr/bin/env python3
"""Bulletproof JSONL sanitize: re-emit each record with ensure_ascii=True so ALL non-ASCII
(incl. U+2028/U+2029/U+0085 line separators that break str.splitlines in the leakage scanner)
is escaped to \\uXXXX. json.loads on read restores the text, so retrieval/scan are unaffected."""
import json
import sys

inp, out = sys.argv[1], sys.argv[2]
n = bad = 0
with open(inp, encoding="utf-8") as f, open(out, "w", encoding="utf-8") as w:
    for ln in f:
        ln = ln.strip()
        if not ln:
            continue
        try:
            d = json.loads(ln)
        except Exception:
            bad += 1
            continue
        w.write(json.dumps(d, ensure_ascii=True) + "\n")
        n += 1
print(f"ascii-sanitized {n} rows ({bad} unparseable dropped) -> {out}")
