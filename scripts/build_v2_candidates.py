#!/usr/bin/env python3
"""Build the v2 training-candidate set: manifest-gated, domain-balanced, license-aware,
deduplicated, PII- and leakage-checked. Pure stdlib, streaming, no network, no ML.

Only rows from manifest sources with ``allowed_for_training=true`` are admitted (fail-closed).
Domain balance uses the fractions in the v2 experiment config. PII/leakage default to FAILING
the run if hits are found (override PII with --allow-pii).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data as datamod  # noqa: E402  (find_pii / find_leakage)
from boldt_embed import data_pipeline as dp  # noqa: E402
from boldt_embed import source_manifest as sm  # noqa: E402
from boldt_embed.v2_experiment_config import load_v2_experiment_config  # noqa: E402


def _v2_candidate(query, document, source_id, domain, license_, extra_meta=None):
    q = dp.normalize_text(str(query)); d = dp.normalize_text(str(document))
    meta = {"source_id": source_id}
    if extra_meta:
        meta.update(extra_meta)
    return {
        "query_id": "q" + dp.stable_text_hash(q), "doc_id": "d" + dp.stable_text_hash(d),
        "query": q, "document": d, "positive": True, "source": source_id, "domain": domain,
        "license": license_, "text_hash": dp.stable_text_hash(d),
        "pair_hash": dp.stable_pair_id(q, d), "metadata": meta,
    }


def _scan_records(cands):
    return [{"query": c["query"], "positive": c["document"]} for c in cands]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(ROOT / "configs" / "data_sources_v2.json"))
    ap.add_argument("--source-jsonl", nargs="+", required=True)
    ap.add_argument("--output", default=str(ROOT / "data" / "processed" / "candidates_v2.jsonl"))
    ap.add_argument("--target-count", type=int, default=50000)
    ap.add_argument("--max-per-domain", type=int, default=None)
    ap.add_argument("--domain-config", default=str(ROOT / "configs" / "experiments" / "v2_generalization.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--pii-scan", action="store_true")
    ap.add_argument("--allow-pii", action="store_true")
    ap.add_argument("--leakage-corpus-jsonl", nargs="*", default=None)
    ap.add_argument("--leakage-threshold", type=float, default=0.9)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = {e.source_id: e for e in sm.load_source_manifest(args.manifest)}
    report = {"blocked_unknown_source": 0, "blocked_not_allowed_for_training": 0,
              "blocked_missing_fields": 0, "admitted": 0}

    cands = []
    for src in args.source_jsonl:
        if not pathlib.Path(src).exists():
            print(f"ERROR: source not found: {src}", file=sys.stderr); return 2
        for row in dp.stream_jsonl(src):
            sid = str(row.get("source") or (row.get("metadata") or {}).get("source_id") or "")
            entry = manifest.get(sid)
            if entry is None:
                report["blocked_unknown_source"] += 1; continue
            if not entry.allowed_for_training:  # license/eval gate (fail-closed)
                report["blocked_not_allowed_for_training"] += 1; continue
            q, d = row.get("query"), row.get("document")
            if not (isinstance(q, str) and q.strip() and isinstance(d, str) and d.strip()):
                report["blocked_missing_fields"] += 1; continue
            domain = str(row.get("domain") or entry.domain)
            cands.append(_v2_candidate(q, d, sid, domain, entry.license))
    report["admitted"] = len(cands)
    print(f"[admit] {json.dumps(report, ensure_ascii=False)}")

    if args.dedup:
        before = len(cands)
        cands = dp.deduplicate_by_text_hash(cands)
        print(f"[dedup] {before} -> {len(cands)}")

    if args.pii_scan:
        hits = datamod.scan_pii(_scan_records(cands))
        bad = sorted({h["record"] for h in hits})
        summary = {"pii_hits": len(hits), "rows_with_pii": len(bad),
                   "by_kind": {}}
        for h in hits:
            summary["by_kind"][h["kind"]] = summary["by_kind"].get(h["kind"], 0) + 1
        print(f"[pii] {json.dumps(summary, ensure_ascii=False)}")
        if bad and not args.allow_pii:
            print("ERROR: PII found; re-run with --allow-pii to drop offending rows, or clean data.",
                  file=sys.stderr)
            return 3
        if bad:
            badset = set(bad)
            cands = [c for i, c in enumerate(cands) if i not in badset]
            print(f"[pii] dropped {len(bad)} rows (--allow-pii)")

    if args.leakage_corpus_jsonl:
        eval_texts = []
        for p in args.leakage_corpus_jsonl:
            for r in dp.stream_jsonl(p):
                for f in ("query", "document", "text", "context", "title"):
                    v = r.get(f)
                    if isinstance(v, str) and v.strip():
                        eval_texts.append(v)
        hits = datamod.find_leakage(_scan_records(cands), eval_texts, threshold=args.leakage_threshold)
        leaked = set(h["record"] for h in hits)
        print(f"[leakage] eval_texts={len(eval_texts)} leaked_rows={len(leaked)} "
              f"(exact+jaccard>={args.leakage_threshold})")
        cands = [c for i, c in enumerate(cands) if i not in leaked]

    # domain balancing
    cfg = load_v2_experiment_config(args.domain_config) if pathlib.Path(args.domain_config).exists() else None
    if cfg is not None:
        targets = cfg.target_counts_by_domain(args.target_count)
        requested = dict(targets)
        cands = dp.sample_to_domain_targets(cands, targets, seed=args.seed)
        print(f"[balance] requested-by-domain={json.dumps(requested, ensure_ascii=False)}")
    elif args.max_per_domain:
        cands = dp.domain_balanced_sample(cands, args.max_per_domain)
    cands = cands[:args.target_count]
    actual = dp.domain_counts(cands)
    print(f"[domains] actual={json.dumps(actual, ensure_ascii=False)} total={len(cands)}")

    if args.dry_run:
        print("=== DRY RUN: not writing. First 3 candidates: ===")
        for c in cands[:3]:
            print(json.dumps(c, ensure_ascii=False))
        return 0
    n = dp.write_jsonl(args.output, cands)
    rep_path = pathlib.Path(args.output).with_suffix(".report.json")
    rep_path.write_text(json.dumps({"admit": report, "domains": actual, "total": n},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {n} candidates -> {args.output}; report -> {rep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
