#!/usr/bin/env python3
"""Fetch German retrieval pairs by DOWNLOADING THE RAW DATA FILES directly (hf_hub_download),
bypassing the script-based loaders that datasets>=4 rejects.

  - mMARCO German (unicamp-dl/mmarco): German collection + train queries + the (qid,pos,neg)
    triples -> real MS-MARCO-grade (query, positive) German pairs. The diversity source.
  - clips/mqa German (data.de.faq / data.de.cqa): German web FAQ/CQA pairs.

Writes candidate format {query, document, source, domain}. Two-pass over the collection so we only
hold the passages our triple-slice needs (the German collection is ~8.8M passages)."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def _hf(repo, fname):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, fname, repo_type="dataset")


def fetch_mmarco(fh, n_triples, seen):
    qf = _hf("unicamp-dl/mmarco", "data/google/queries/train/german_queries.train.tsv")
    tf = _hf("unicamp-dl/mmarco", "data/triples.train.ids.small.tsv")
    print(f"[mmarco] queries={qf}\n[mmarco] triples={tf}", flush=True)
    queries = {}
    with open(qf, encoding="utf-8") as f:
        for ln in f:
            qid, _, q = ln.partition("\t")
            queries[qid.strip()] = q.strip()
    print(f"[mmarco] loaded {len(queries)} train queries", flush=True)
    # pass 1: take first n_triples (qid,pos_pid); collect needed pids
    want = []
    need_pids = set()
    with open(tf, encoding="utf-8") as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            qid, pos = p[0].strip(), p[1].strip()
            if qid in queries:
                want.append((qid, pos)); need_pids.add(pos)
            if len(want) >= n_triples:
                break
    print(f"[mmarco] {len(want)} (qid,pos) wanted; resolving {len(need_pids)} passages", flush=True)
    cf = _hf("unicamp-dl/mmarco", "data/google/collections/german_collection.tsv")
    coll = {}
    with open(cf, encoding="utf-8") as f:
        for ln in f:
            pid, _, txt = ln.partition("\t")
            pid = pid.strip()
            if pid in need_pids:
                coll[pid] = txt.strip()
    written = 0
    for qid, pos in want:
        q, d = queries.get(qid, ""), coll.get(pos, "")
        if q and len(d) > 5:
            k = (q[:200], d[:200])
            if k not in seen:
                seen.add(k)
                fh.write(json.dumps({"query": q, "document": d, "source": "mmarco_de",
                                     "domain": "msmarco_web"}, ensure_ascii=False) + "\n")
                written += 1
    print(f"[mmarco] wrote {written} pairs", flush=True)
    return written


def fetch_mqa(fh, seen, cap=120000):
    written = 0
    for kind in ("faq", "cqa"):
        try:
            fp = _hf("clips/mqa", f"data/data.de.{kind}.json.gz")
        except Exception as e:
            print(f"[mqa] {kind} ERR {type(e).__name__}: {str(e)[:80]}", flush=True); continue
        op = gzip.open(fp, "rt", encoding="utf-8")
        first = op.readline(); op.seek(0)
        rows = []
        try:  # the file is JSON-lines OR a JSON array
            json.loads(first); rows = (json.loads(l) for l in op)
        except Exception:
            op.seek(0); rows = json.load(op)
        c = 0
        for r in rows:
            q = (r.get("name") or r.get("title") or r.get("question") or "").strip()
            ans = r.get("answers") or r.get("text") or r.get("answer")
            if isinstance(ans, list) and ans:
                a0 = ans[0]
                d = (a0.get("text") if isinstance(a0, dict) else a0) or ""
            else:
                d = ans or ""
            d = (d if isinstance(d, str) else "").strip()
            if q and len(d) > 5:
                k = (q[:200], d[:200])
                if k not in seen:
                    seen.add(k)
                    fh.write(json.dumps({"query": q, "document": d, "source": f"mqa_de_{kind}",
                                         "domain": "faq_web"}, ensure_ascii=False) + "\n")
                    written += 1; c += 1
            if c >= cap:
                break
        print(f"[mqa] {kind}: wrote {c}", flush=True)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="outputs/v8/data/mmarco_mqa_de.jsonl")
    ap.add_argument("--mmarco-triples", type=int, default=400000)
    ap.add_argument("--skip-mqa", action="store_true")
    args = ap.parse_args()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    seen = set(); counts = {}
    with out.open("w", encoding="utf-8") as fh:
        try:
            counts["mmarco_de"] = fetch_mmarco(fh, args.mmarco_triples, seen)
        except Exception as e:
            counts["mmarco_de"] = f"ERR {type(e).__name__}: {e}"
        if not args.skip_mqa:
            try:
                counts["mqa_de"] = fetch_mqa(fh, seen)
            except Exception as e:
                counts["mqa_de"] = f"ERR {type(e).__name__}: {e}"
    print(json.dumps({"output": str(out), "total_unique": len(seen), "per_source": counts},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
