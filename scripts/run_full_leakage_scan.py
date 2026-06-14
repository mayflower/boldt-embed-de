#!/usr/bin/env python3
"""Full leakage scan: ALL training candidates vs ALL held-out eval corpora (scalable, stdlib).

Replaces the v2 O(n*m) subset scan. Builds (or loads) a blocking index over the eval corpora,
then two-stage scans the candidates (block -> exact-verify). Writes a JSON report and a hits
JSONL; ``--drop-hits`` also writes a cleaned candidate file. This is the v3 pre-training gate:
training on v3 candidates requires a clean (or cleaned) report (see
``leakage_index.require_clean_leakage_report``).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import leakage_index as li  # noqa: E402
from build_leakage_index import _iter_eval_units  # noqa: E402

DEFAULT_OUT = ROOT / "outputs" / "v3-real-domain" / "leakage" / "leakage_report.json"
DEFAULT_HITS = ROOT / "outputs" / "v3-real-domain" / "leakage" / "leakage_hits.jsonl"


def _read_jsonl(path):
    return [json.loads(l) for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _counts(hits, key):
    c = {}
    for h in hits:
        v = str(h.get(key) or "unknown")
        c[v] = c.get(v, 0) + 1
    return dict(sorted(c.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", required=True, help="candidate JSONL")
    ap.add_argument("--eval-corpus", nargs="*", default=None, help="eval JSONL(s) or dataset=path")
    ap.add_argument("--index", default=None, help="prebuilt index JSON (instead of --eval-corpus)")
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--hits-output", default=str(DEFAULT_HITS))
    ap.add_argument("--drop-hits", default=None, help="write a cleaned candidate JSONL here")
    ap.add_argument("--shingle-n", type=int, default=li.DEFAULT_SHINGLE_N)
    ap.add_argument("--num-perm", type=int, default=li.DEFAULT_NUM_PERM)
    ap.add_argument("--jaccard-threshold", type=float, default=li.DEFAULT_JACCARD_THRESHOLD)
    args = ap.parse_args()

    if not args.index and not args.eval_corpus:
        print("ERROR: pass --eval-corpus or --index", file=sys.stderr)
        return 2

    t0 = time.monotonic()
    if args.index:
        index = li.LeakageIndex.from_dict(json.loads(pathlib.Path(args.index).read_text("utf-8")))
    else:
        index = li.build_eval_leakage_index(_iter_eval_units(args.eval_corpus),
                                            shingle_n=args.shingle_n, num_perm=args.num_perm)
    candidates = _read_jsonl(args.candidates)
    result = li.find_candidate_leakage(candidates, index, jaccard_threshold=args.jaccard_threshold)
    hits = result["hits"]
    runtime = round(time.monotonic() - t0, 4)

    by_kind = _counts(hits, "kind")
    dropped_ids = sorted({h["candidate_id"] for h in hits})
    report = {
        "status": "ok",
        "candidates_path": str(args.candidates),
        "n_train_candidates": len(candidates),
        "n_eval_texts": index.n_eval_texts(),
        "exact_hits": by_kind.get("exact", 0),
        "exact_normalized_hits": by_kind.get("exact_normalized", 0),
        "near_duplicate_hits": by_kind.get("near_duplicate", 0),
        "total_flagged_candidates": len(dropped_ids),
        "hits_by_eval_dataset": _counts(hits, "eval_dataset"),
        "hits_by_source": _counts(hits, "source"),
        "hits_by_domain": _counts(hits, "domain"),
        "hits_by_license": _counts(hits, "license"),
        "dropped_candidate_ids": dropped_ids,
        "jaccard_threshold": args.jaccard_threshold,
        "scan_runtime_sec": runtime,
        "jaccard_comparisons": result["stats"]["jaccard_comparisons"],
        "blocked_pairs": result["stats"]["blocked_pairs"],
        "naive_comparisons": len(candidates) * index.n_eval_texts(),
    }

    # write hits + report
    pathlib.Path(args.hits_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.hits_output, "w", encoding="utf-8") as f:
        for h in hits:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    if args.drop_hits:
        drop = set(dropped_ids)
        kept = [c for c in candidates if li._candidate_id(c) not in drop]
        pathlib.Path(args.drop_hits).parent.mkdir(parents=True, exist_ok=True)
        with open(args.drop_hits, "w", encoding="utf-8") as f:
            for c in kept:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        report["cleaned_candidates_path"] = str(args.drop_hits)
        report["n_candidates_after_drop"] = len(kept)
        print(f"[clean] dropped {len(candidates) - len(kept)} -> {args.drop_hits} "
              f"({len(kept)} kept)")
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    print(f"[leakage] {len(candidates)} candidates x {index.n_eval_texts()} eval texts: "
          f"{report['exact_hits']} exact, {report['exact_normalized_hits']} exact-norm, "
          f"{report['near_duplicate_hits']} near-dup; "
          f"{report['jaccard_comparisons']} verify-comparisons "
          f"(naive would be {report['naive_comparisons']}); {runtime}s -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
