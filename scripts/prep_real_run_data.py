#!/usr/bin/env python3
"""Prepare REAL data for an executed teacher/student run (network, no GPU).

Writes, under data/processed/ (git-ignored):
  raw_sources.jsonl     - non-benchmark training rows {query, document, domain, license, source}
                          from deutsche-telekom/wikipedia-22-12-de-dpr (train) + clips/mqa (de).
  eval_leakage.jsonl    - eval-corpus texts (GermanQuAD test + GerDaLIR) to filter training against.
  eval/<name>_{corpus,queries,qrels}.jsonl - held-out eval fixtures (gerdalir, germanquad, dt_test).

Public benchmark test data is used ONLY for evaluation + leakage filtering, never as training.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
EVAL = OUT / "eval"


def _w(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):6d} -> {path.relative_to(ROOT)}")
    return len(rows)


def dt_train_rows(n):
    from datasets import load_dataset
    ds = load_dataset("deutsche-telekom/wikipedia-22-12-de-dpr")["train"]
    rows = []
    for row in ds:
        ctx = row["context"]
        qs = list(row.get("question") or [])
        for q in qs[:1]:
            if q and q.strip():
                rows.append({"query": q, "document": ctx, "domain": "wiki",
                             "license": "CC-BY-SA-4.0", "source": "DT-de-dpr"})
        if len(rows) >= n:
            break
    return rows[:n]


def paraphrase_rows(n):
    """deutsche-telekom/ger-backtrans-paraphrase: German paraphrase pairs (de <-> en_de)
    drawn from MULTIPLE non-Wikipedia corpora (TED, news, Europarl, ...) -> real domain
    diversity. (sentence, paraphrase) is a valid contrastive positive. Streamed + capped."""
    from datasets import load_dataset
    rows = []
    try:
        ds = load_dataset("deutsche-telekom/ger-backtrans-paraphrase", split="train", streaming=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [paraphrase] could not load: {exc}")
        return rows
    for ex in ds:
        de, en_de = (ex.get("de") or "").strip(), (ex.get("en_de") or "").strip()
        corpus = str(ex.get("corpus") or "misc").lower()
        if corpus == "wikipedia":
            continue  # keep this source strictly non-wiki
        if de and en_de and de != en_de and len(de) > 25 and len(en_de) > 25:
            rows.append({"query": de, "document": en_de, "domain": f"para_{corpus}",
                         "license": "CC-BY-SA-4.0", "source": "ger-backtrans-paraphrase"})
        if len(rows) >= n:
            break
    return rows


def swim_rows(n):
    """nthakur/swim-ir-monolingual (de): synthetic query -> passage (CC-BY-SA). Wikipedia-
    derived, but a different (synthetic) query style than DT -> query-style diversity."""
    from datasets import load_dataset
    rows = []
    try:
        ds = load_dataset("nthakur/swim-ir-monolingual", "de", split="train", streaming=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [swim] could not load: {exc}")
        return rows
    for ex in ds:
        q, text = (ex.get("query") or "").strip(), (ex.get("text") or "").strip()
        if q and text and len(text) > 40:
            rows.append({"query": q, "document": text, "domain": "wiki_synth",
                         "license": "CC-BY-SA-4.0", "source": "swim-ir-de"})
        if len(rows) >= n:
            break
    return rows


def load_gerdalir():
    from datasets import load_dataset
    c = load_dataset("mteb/GerDaLIRSmall", "corpus")["corpus"]
    q = load_dataset("mteb/GerDaLIRSmall", "queries")["queries"]
    rel = load_dataset("mteb/GerDaLIRSmall", "default")["test"]
    qrels = {}
    for r in rel:
        if float(r["score"]) > 0:
            qrels.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
    corpus = [{"doc_id": str(r["_id"]), "text": ((r.get("title") or "") + " " + r["text"]).strip()}
              for r in c]
    queries, qrels_rows = [], []
    for x in q:
        qid = str(x["_id"])
        if qid in qrels:
            queries.append({"query_id": qid, "query": x["text"]})
            for did in qrels[qid]:
                qrels_rows.append({"query_id": qid, "doc_id": did, "relevance": 1})
    return corpus, queries, qrels_rows


def load_germanquad(n_q):
    from datasets import load_dataset
    base = ("https://huggingface.co/datasets/deepset/germanquad/resolve/"
            "refs%2Fconvert%2Fparquet/plain_text")
    test = load_dataset("parquet", data_files={"test": f"{base}/test/0000.parquet"})["test"]
    cidx, corpus, queries, qrels = {}, [], [], []
    for i, ex in enumerate(test):
        c = ex["context"]
        if c not in cidx:
            cidx[c] = f"g{len(corpus)}"
            corpus.append({"doc_id": cidx[c], "text": c})
        qid = f"gq{i}"
        queries.append({"query_id": qid, "query": ex["question"]})
        qrels.append({"query_id": qid, "doc_id": cidx[c], "relevance": 1})
    random.Random(0).shuffle(queries)
    keep = {q["query_id"] for q in queries[:n_q]}
    queries = [q for q in queries if q["query_id"] in keep]
    qrels = [r for r in qrels if r["query_id"] in keep]
    return corpus, queries, qrels


def load_dt_test(n_q):
    from datasets import load_dataset
    test = load_dataset("deutsche-telekom/wikipedia-22-12-de-dpr")["test"]
    cidx, corpus, queries, qrels = {}, [], [], []
    for i, row in enumerate(test):
        c = row["context"]
        if c not in cidx:
            cidx[c] = f"d{len(corpus)}"
            corpus.append({"doc_id": cidx[c], "text": c})
        qs = list(row.get("question") or [])
        if qs:
            qid = f"dt{i}"
            queries.append({"query_id": qid, "query": qs[0]})
            qrels.append({"query_id": qid, "doc_id": cidx[c], "relevance": 1})
    random.Random(0).shuffle(queries)
    keep = {q["query_id"] for q in queries[:n_q]}
    queries = [q for q in queries if q["query_id"] in keep]
    qrels = [r for r in qrels if r["query_id"] in keep]
    return corpus, queries, qrels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dt-rows", type=int, default=3000)
    ap.add_argument("--para-rows", type=int, default=4000)
    ap.add_argument("--swim-rows", type=int, default=2000)
    ap.add_argument("--gq-queries", type=int, default=1500)
    ap.add_argument("--dt-queries", type=int, default=1000)
    args = ap.parse_args()

    print("=== training sources (non-benchmark, multi-domain) ===")
    raw = dt_train_rows(args.dt_rows)
    print(f"  DT-de-dpr (wiki): {len(raw)}")
    para = paraphrase_rows(args.para_rows)
    print(f"  ger-backtrans-paraphrase (TED/news/Europarl/...): {len(para)}")
    swim = swim_rows(args.swim_rows)
    print(f"  swim-ir de (wiki synthetic): {len(swim)}")
    raw += para + swim
    random.Random(0).shuffle(raw)
    _w(OUT / "raw_sources.jsonl", raw)

    print("=== held-out eval fixtures ===")
    leak = []
    for name, (corpus, queries, qrels) in {
        "gerdalir": load_gerdalir(),
        "germanquad": load_germanquad(args.gq_queries),
        "dt_test": load_dt_test(args.dt_queries),
    }.items():
        _w(EVAL / f"{name}_corpus.jsonl", corpus)
        _w(EVAL / f"{name}_queries.jsonl", queries)
        _w(EVAL / f"{name}_qrels.jsonl", qrels)
        leak += [{"text": c["text"]} for c in corpus]
        leak += [{"text": q["query"]} for q in queries]
    _w(OUT / "eval_leakage.jsonl", leak)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
