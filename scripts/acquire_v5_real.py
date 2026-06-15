#!/usr/bin/env python3
"""Acquire REAL multi-domain German RAG training data for v5 (no LLM, no fabrication).

Sources (real, on disk / cached):
- faq_real: data/raw/v3/faq_real_local.jsonl (real WebFAQ, CC-BY-4.0)
- qa_passage_non_eval / german_stress / long_doc_chunks: deutsche-telekom/wikipedia-22-12-de-dpr
  TRAIN split (real Wikipedia passages + real German questions, CC-BY-SA-4.0). The natural
  `question` -> qa_passage_non_eval; the `imperative_formal` reformulation -> german_stress;
  long contexts -> long_doc_chunks.

LEAKAGE-SAFE: every DPR-train context is dropped if its normalized text matches a guardrail
passage (dt_test = DPR *test* split; GermanQuAD). Dropped counts are reported. Queries are REAL
(synthetic_query=false), so no teacher-validation gate is needed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().lower())[:400]


def _read_jsonl(path: pathlib.Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]


def _leakage_texts() -> set:
    leak = set()
    dt = ROOT / "data/processed/eval/dt_test_corpus.jsonl"
    if dt.exists():
        for r in _read_jsonl(dt):
            leak.add(_norm(r.get("text", "")))
    gq = ROOT / "data/processed/eval/germanquad_corpus.jsonl"
    if gq.exists():
        for r in _read_jsonl(gq):
            for k in ("text", "document", "passage"):
                if r.get(k):
                    leak.add(_norm(r[k]))
    return {t for t in leak if t}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--faq", default=str(ROOT / "data/raw/v3/faq_real_local.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/raw/v5"))
    ap.add_argument("--max-faq", type=int, default=2000)
    ap.add_argument("--max-qa", type=int, default=2500)
    ap.add_argument("--max-stress", type=int, default=1200)
    ap.add_argument("--max-longdoc", type=int, default=800)
    ap.add_argument("--longdoc-min-chars", type=int, default=1200)
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    leak = _leakage_texts()
    print(f"[acquire-v5] leakage guard: {len(leak)} guardrail passages (dt_test + germanquad)")

    from datasets import load_dataset
    ds = load_dataset("deutsche-telekom/wikipedia-22-12-de-dpr")["train"]

    qa, stress, longdoc = [], [], []
    dropped_leak = 0
    seen_ctx = set()
    for r in ds:
        ctx = (r.get("context") or "").strip()
        if not ctx:
            continue
        n = _norm(ctx)
        if n in leak:
            dropped_leak += 1
            continue
        if n in seen_ctx:
            continue
        seen_ctx.add(n)
        uuid = r.get("context_uuid") or str(r.get("wiki_id"))
        title = r.get("title") or ""
        url = f"https://de.wikipedia.org/?curid={r.get('wiki_id')}" if r.get("wiki_id") else None
        questions = r.get("question") or []
        imperatives = r.get("imperative_formal") or []

        def _row(domain, query):
            row = {"source_id": f"dtdpr-{uuid}", "domain": domain, "query": query,
                   "document": ctx, "title": title, "license": "CC-BY-SA-4.0",
                   "synthetic_query": False, "eval_only": False, "public_benchmark": False}
            if url:
                row["source_url"] = url
            return row

        if questions and len(qa) < args.max_qa:
            qa.append(_row("qa_passage_non_eval", questions[0]))
        if imperatives and len(stress) < args.max_stress:
            stress.append(_row("german_stress", imperatives[0]))
        if len(ctx) >= args.longdoc_min_chars and len(longdoc) < args.max_longdoc:
            q = questions[1] if len(questions) > 1 else (questions[0] if questions else None)
            if q:
                longdoc.append(_row("long_doc_chunks", q))
        if (len(qa) >= args.max_qa and len(stress) >= args.max_stress
                and len(longdoc) >= args.max_longdoc):
            break

    # faq_real (real WebFAQ)
    faq_rows = _read_jsonl(pathlib.Path(args.faq))[:args.max_faq]
    faq = [{"source_id": f"webfaq-{i}", "domain": "faq_real", "query": r.get("query", ""),
            "document": r.get("document", ""), "title": r.get("title", ""),
            "license": r.get("license", "CC-BY-4.0"), "synthetic_query": False,
            "eval_only": False, "public_benchmark": False,
            **({"source_url": r["url"]} if r.get("url") else {})}
           for i, r in enumerate(faq_rows)]

    written = {}
    for name, rows in (("faq_real", faq), ("qa_passage_non_eval", qa),
                       ("german_stress", stress), ("long_doc_chunks", longdoc)):
        p = out / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        written[name] = len(rows)

    total = sum(written.values())
    report = {"written_by_domain": written, "total_rows": total,
              "dropped_leakage_contexts": dropped_leak,
              "faq_share": round(written["faq_real"] / total, 4) if total else 0,
              "domains_populated": [k for k, v in written.items() if v > 0],
              "domains_missing_no_real_source": ["web_nonfaq", "local_rag"]}
    (out / "acquire_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(f"[acquire-v5] wrote {total} REAL rows: {written}")
    print(f"[acquire-v5] dropped {dropped_leak} leakage contexts; faq_share={report['faq_share']}")
    print(f"[acquire-v5] domains populated: {report['domains_populated']}; "
          f"missing (no real source, NOT faked): web_nonfaq, local_rag")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
