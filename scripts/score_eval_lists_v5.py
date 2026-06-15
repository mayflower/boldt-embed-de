#!/usr/bin/env python3
"""Score fixed eval candidate lists with a reranker ONCE (model loaded once, all pairs batched),
writing reranker_score onto each candidate. Then eval_v5_rag_lift can run with no ML reloads.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _read(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reranker", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "training_reranker.json"))
    ap.add_argument("--lists", nargs="+", required=True, help="name=path fixed candidate lists")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from boldt_embed import reranker_modern as RM
    from boldt_embed.config import load_reranker_config
    tmpl = load_reranker_config(args.config).input_template
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for spec in args.lists:
        name, path = spec.split("=", 1)
        rows = _read(path)
        # flatten ALL pairs for this set -> one batched scoring call (model loaded once per set)
        flat, index = [], []
        for ri, r in enumerate(rows):
            for ci, c in enumerate(r.get("candidates") or []):
                flat.append((r.get("query", ""), c.get("text") or c.get("document", "")))
                index.append((ri, ci))
        scores = RM.score_with_student_reranker(args.reranker, flat, tmpl, max_length=256)
        for (ri, ci), s in zip(index, scores):
            rows[ri]["candidates"][ci]["reranker_score"] = float(s)
        outp = out / f"{name}_scored.jsonl"
        with outp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[score-eval] {name}: {len(rows)} lists, {len(flat)} pairs scored -> {outp}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
