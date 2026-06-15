#!/usr/bin/env python3
"""Prep v5 multi-domain RAG data (data/raw/v5/*.jsonl) into corpus/queries/qrels for BM25 +
candidate-list building, then attach the per-query domain back onto the built candidate lists.

Two modes:
  --stage prep    : write {corpus,queries,qrels}.jsonl from data/raw/v5 (queries carry domain)
  --stage attach  : join domain onto a built candidate-list JSONL (by query_id)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]


def _did(text):
    return "d" + hashlib.blake2b(text.encode("utf-8"), digest_size=10).hexdigest()


def _qid(q, sid):
    return "q" + hashlib.blake2b(f"{sid}\x1f{q}".encode("utf-8"), digest_size=10).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "attach"], required=True)
    ap.add_argument("--raw-dir", default=str(ROOT / "data/raw/v5"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/v5-small-rag/train"))
    ap.add_argument("--candidate-lists", default=None)
    ap.add_argument("--domain-map", default=None)
    args = ap.parse_args()

    if args.stage == "prep":
        rows = []
        for f in sorted(pathlib.Path(args.raw_dir).glob("*.jsonl")):
            if f.name == "acquire_report.json":
                continue
            rows += _read(f)
        corpus, queries, qrels, dmap = {}, [], [], {}
        for r in rows:
            doc = r["document"]
            did = _did(doc)
            corpus.setdefault(did, {"doc_id": did, "text": doc})
            qid = _qid(r["query"], r["source_id"])
            queries.append({"query_id": qid, "query": r["query"], "domain": r["domain"]})
            qrels.append({"query_id": qid, "doc_id": did, "relevance": 1})
            dmap[qid] = r["domain"]
        out = pathlib.Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, data in (("corpus", list(corpus.values())), ("queries", queries),
                           ("qrels", qrels)):
            with (out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
                for x in data:
                    fh.write(json.dumps(x, ensure_ascii=False) + "\n")
        (out / "domain_map.json").write_text(json.dumps(dmap, ensure_ascii=False), encoding="utf-8")
        print(f"[prep] {len(queries)} queries, {len(corpus)} docs -> {out}")
        return 0

    # attach domain to built candidate lists
    dmap = json.loads(pathlib.Path(args.domain_map).read_text(encoding="utf-8"))
    p = pathlib.Path(args.candidate_lists)
    rows = _read(p)
    for r in rows:
        r["domain"] = dmap.get(str(r.get("query_id")), "unknown")
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_dom = {}
    for r in rows:
        by_dom[r["domain"]] = by_dom.get(r["domain"], 0) + 1
    print(f"[attach] domain attached to {len(rows)} lists: {by_dom}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
