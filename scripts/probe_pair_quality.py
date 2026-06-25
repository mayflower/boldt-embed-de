#!/usr/bin/env python3
"""Per-source training-PAIR quality probe: mean query/positive alignment via an INDEPENDENT model
(multilingual-e5-base, not the model under test). Low alignment = noisy/mislabeled pairs.

Grounds the data-quality question with a measurement instead of eyeballing. Needs the [eval] extra."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SOURCES = {
    "v6_clean(train)": "outputs/autoresearch/prepared/train_candidates.clean.jsonl",
    "swim_ir_de(wiki)": "data/raw/v2/swim_ir_de.jsonl",
    "dt_de_dpr(wiki)": "data/raw/v3/dt_de_dpr.jsonl",
    "faq_real_local": "data/raw/v3/faq_real_local.jsonl",
    "ger_backtrans": "data/raw/v3/ger_backtrans_paraphrase.jsonl",
    "synthetic_faq_v2": "data/raw/v2/synthetic_faq_v2.jsonl",
    "synthetic_legal_v2": "data/raw/v2/synthetic_legal_v2.jsonl",
    "synthetic_cross_lingual_v2": "data/raw/v2/synthetic_cross_lingual_v2.jsonl",
    "local_admin_v2": "data/raw/v2/local_admin_v2.jsonl",
    "candidates_v2(union)": "data/processed/candidates_v2.jsonl",
}


def _s(x):
    if isinstance(x, list):
        x = x[0] if x else ""
    return x.strip() if isinstance(x, str) else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=400, help="pairs sampled per source")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    from sentence_transformers import SentenceTransformer
    import numpy as np
    m = SentenceTransformer("intfloat/multilingual-e5-base")  # independent judge

    print(f"{'source':<28}{'n':>5}{'mean_cos':>10}{'<0.70':>8}{'<0.60':>8}")
    print("-" * 59)
    for name, path in SOURCES.items():
        p = Path(path)
        if not p.exists():
            print(f"{name:<28}  MISSING"); continue
        rows = []
        for ln in p.open(encoding="utf-8"):
            if ln.strip():
                try: rows.append(json.loads(ln))
                except Exception: pass
        rng.shuffle(rows)
        qs, ds = [], []
        for r in rows:
            q = _s(r.get("query"))
            d = _s(r.get("positive")) or _s(r.get("document"))
            if isinstance(r.get("positive"), bool):  # candidates_v2: positive is a flag
                d = _s(r.get("document"))
            if q and d and len(d) > 5:
                qs.append("query: " + q); ds.append("passage: " + d)
            if len(qs) >= args.n:
                break
        if not qs:
            print(f"{name:<28}  no usable pairs"); continue
        eq = m.encode(qs, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        ed = m.encode(ds, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        cos = (eq * ed).sum(1)
        print(f"{name:<28}{len(qs):>5}{float(cos.mean()):>10.3f}"
              f"{float((cos < 0.70).mean()):>8.2f}{float((cos < 0.60).mean()):>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
