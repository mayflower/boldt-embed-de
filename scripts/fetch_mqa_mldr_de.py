#!/usr/bin/env python3
"""Fetch the two stragglers by direct file download (bypass datasets>=4 script block):
  - clips/mqa German: nested web-FAQ pages -> (question 'name', answer 'text') pairs, dropping the
    frequent null answers (that's why the naive parse got 0).
  - Shitao/MLDR German TRAIN split: long-doc (query -> positive passage). FETCH + ASSESS ONLY;
    MLDR is a benchmark we eval on, so this output must be leakage-scanned vs the MLDR eval and is
    NOT training_usable until then.
Writes {query, document, source, domain} to separate files so MLDR stays quarantined."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def _hf(repo, fname):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, fname, repo_type="dataset")


def fetch_mqa(out_path):
    seen = set()
    n = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for kind in ("faq", "cqa"):
            try:
                fp = _hf("clips/mqa", f"data/data.de.{kind}.json.gz")
            except Exception as e:
                print(f"[mqa] {kind} ERR {type(e).__name__}: {str(e)[:80]}", flush=True); continue
            c = 0
            for ln in gzip.open(fp, "rt", encoding="utf-8"):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                qs = rec.get("questions") or []
                if isinstance(qs, str):
                    try: qs = eval(qs)  # noqa: S307 - local trusted
                    except Exception: qs = []
                for q in qs:
                    if not isinstance(q, dict):
                        continue
                    name = (q.get("name") or "").strip()
                    ans = q.get("text")
                    # answer may be under 'answers'/'accepted_answer'
                    if not ans:
                        a = q.get("answers") or q.get("accepted_answer")
                        if isinstance(a, list) and a:
                            ans = (a[0].get("text") if isinstance(a[0], dict) else a[0])
                        elif isinstance(a, dict):
                            ans = a.get("text")
                    ans = (ans or "").strip() if isinstance(ans, str) else ""
                    if name and len(ans) > 5:
                        k = (name[:200], ans[:200])
                        if k not in seen:
                            seen.add(k)
                            w.write(json.dumps({"query": name, "document": ans,
                                                "source": f"mqa_de_{kind}", "domain": "faq_web"},
                                               ensure_ascii=False) + "\n")
                            n += 1; c += 1
            print(f"[mqa] {kind}: wrote {c}", flush=True)
    return n


def fetch_mldr(out_path):
    fp = _hf("Shitao/MLDR", "mldr-v1.0-de/train.jsonl.gz")
    n = 0
    keys_seen = None
    with open(out_path, "w", encoding="utf-8") as w:
        for ln in gzip.open(fp, "rt", encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if keys_seen is None:
                keys_seen = list(r.keys()); print(f"[mldr] record keys: {keys_seen}", flush=True)
            q = (r.get("query") or r.get("question") or "").strip()
            pos = r.get("positive_passages") or r.get("positive_ctxs") or r.get("pos")
            txt = ""
            if isinstance(pos, list) and pos:
                p0 = pos[0]
                txt = (p0.get("text") if isinstance(p0, dict) else p0) or ""
            txt = txt.strip() if isinstance(txt, str) else ""
            if q and len(txt) > 5:
                w.write(json.dumps({"query": q, "document": txt, "source": "mldr_de_train",
                                    "domain": "long_doc_BENCHMARK"}, ensure_ascii=False) + "\n")
                n += 1
    print(f"[mldr] wrote {n} (QUARANTINED: benchmark train split, scan vs MLDR eval before use)",
          flush=True)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mqa-out", default="outputs/v8/data/mqa_de_raw.jsonl")
    ap.add_argument("--mldr-out", default="outputs/v8/data/mldr_de_train_raw.jsonl")
    args = ap.parse_args()
    Path(args.mqa_out).parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    try:
        counts["mqa_de"] = fetch_mqa(args.mqa_out)
    except Exception as e:
        counts["mqa_de"] = f"ERR {type(e).__name__}: {e}"
    try:
        counts["mldr_de_train"] = fetch_mldr(args.mldr_out)
    except Exception as e:
        counts["mldr_de_train"] = f"ERR {type(e).__name__}: {e}"
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
