#!/usr/bin/env python3
"""Build the v3 candidate set from REAL domain corpora — strict quotas, full provenance, safety.

Replaces build_v2_candidates for the next experiment. Reads the v3 source manifest + the v3
experiment config + the materialized raw drops (data/raw/v3/<source_id>.jsonl). For each
training-allowed source it emits fully-provenanced (query, document) candidates, OR — for
document-only corpora — passage records (NEVER fabricating a query/doc pair). It deduplicates,
PII-scans, leakage-filters against the eval index, then samples to the per-domain quotas and
reports achieved-vs-target with a real-vs-synthetic split.

Pure stdlib, no ML, no network.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data as datamod  # noqa: E402  (scan_pii)
from boldt_embed import data_pipeline as dp  # noqa: E402
from boldt_embed import domain_source_acquisition as dsa  # noqa: E402
from boldt_embed import leakage_index as li  # noqa: E402

REAL_DOMAINS = ("faq_real", "admin_real", "legal_adjacency_real_no_eval_overlap")


def _provenance(entry: dsa.V3SourceEntry, row: dict) -> dict:
    """Derive v3 provenance from the manifest entry (authoritative) + the raw row.
    Synthetic/inherited sources keep a concrete inherited license when the row carries one."""
    origin = "inherited" if (entry.supplemental or "inherit" in entry.license.lower()) else "manifest"
    if origin == "inherited":
        rl = row.get("license")
        license_ = rl if isinstance(rl, str) and rl.strip() and "inherit" not in rl.lower() else entry.license
    else:
        license_ = entry.license
    synthetic = bool(entry.supplemental or origin == "inherited"
                     or row.get("synthetic") or row.get("generated"))
    return {"source_id": entry.source_id, "domain": entry.domain, "license": license_,
            "license_origin": origin, "allowed_for_training": entry.allowed_for_training,
            "synthetic": synthetic,
            "source_url": row.get("url") or entry.raw.get("source_url") or None}


def _raw_path(entry: dsa.V3SourceEntry, raw_dir: pathlib.Path) -> pathlib.Path:
    """Prefer raw_dir/<source_id>.jsonl (acquire output); fall back to the manifest loader path."""
    p = raw_dir / f"{entry.source_id}.jsonl"
    if p.exists():
        return p
    lp = entry.loader_path()
    return pathlib.Path(lp) if lp else p


def build(entries, raw_dir, targets, *, leakage_index=None, pii_scan=True,
          real_domains=REAL_DOMAINS, seed=0) -> dict:
    raw_dir = pathlib.Path(raw_dir)
    dropped = {"blocked_source": 0, "no_data": 0, "missing_fields": 0,
               "unknown_license": 0, "dedup": 0, "pii": 0, "leakage": 0}
    blocked_sources, candidates, passages = [], [], []

    for e in entries:
        if not e.allowed_for_training:
            reason = ("public_benchmark" if e.public_benchmark else
                      "license_unverified" if not e.license_verified else
                      "eval_only" if e.eval_only else
                      "overlap_risk" if e.contains_eval_overlap_risk else "not_allowed")
            blocked_sources.append({"source_id": e.source_id, "domain": e.domain, "reason": reason})
            dropped["blocked_source"] += 1
            continue
        path = _raw_path(e, raw_dir)
        if not path.exists():
            blocked_sources.append({"source_id": e.source_id, "domain": e.domain,
                                    "reason": f"no_data:{path}"})
            dropped["no_data"] += 1
            continue
        document_only = e.source_type == "local_corpus_jsonl"
        for row in dp.stream_jsonl(str(path)):
            prov = _provenance(e, row)
            if li_is_unknown(prov["license"]):
                dropped["unknown_license"] += 1
                continue
            if document_only:
                doc = row.get("text") or row.get("document")
                if not (isinstance(doc, str) and doc.strip()):
                    dropped["missing_fields"] += 1
                    continue
                passages.append(dp.build_v3_passage_record(doc, **prov))
            else:
                q, d = row.get("query"), (row.get("document") or row.get("text"))
                if not (isinstance(q, str) and q.strip() and isinstance(d, str) and d.strip()):
                    dropped["missing_fields"] += 1
                    continue
                candidates.append(dp.build_v3_candidate_row(q, d, **prov))

    # dedup pairs by (query, document)
    before = len(candidates)
    candidates = dp.deduplicate_by_text_hash(candidates)
    dropped["dedup"] += before - len(candidates)

    if pii_scan and candidates:
        hits = datamod.scan_pii([{"query": c["query"], "positive": c["document"]} for c in candidates])
        bad = {h["record"] for h in hits}
        dropped["pii"] += len(bad)
        candidates = [c for i, c in enumerate(candidates) if i not in bad]

    if leakage_index is not None and candidates:
        res = li.find_candidate_leakage(candidates, leakage_index)
        hit_ids = {h["candidate_id"] for h in res["hits"]}
        before = len(candidates)
        candidates = [c for c in candidates if c["pair_hash"] not in hit_ids]
        dropped["leakage"] += before - len(candidates)

    sampled = dp.sample_to_domain_targets(candidates, targets, seed=seed)
    quota = dp.quota_report(sampled, targets, real_domains=real_domains)
    report = {
        "totals": {"admitted_pairs": len(candidates), "selected_pairs": len(sampled),
                   "passages": len(passages),
                   "real_domains_with_pairs": sorted(
                       {c["domain"] for c in sampled if c["domain"] in real_domains
                        and not c["synthetic"]})},
        "dropped_by_reason": dropped,
        "blocked_sources": blocked_sources,
        "quota": quota,
        "domain_counts_selected": dp.domain_counts(sampled),
        "passages_by_domain": dp.domain_counts(passages),
    }
    return {"candidates": sampled, "passages": passages, "report": report}


def li_is_unknown(v) -> bool:
    s = (str(v).strip().lower() if v is not None else "")
    return s == "" or s == "unknown"


def render_markdown(report: dict, status: str) -> str:
    t = report["totals"]
    lines = ["# v3 candidate build", "", f"Status: **{status.upper()}**", "",
             f"- selected pairs: {t['selected_pairs']} (admitted {t['admitted_pairs']}) · "
             f"passages: {t['passages']}",
             f"- real domains with real pairs: {t['real_domains_with_pairs']}", "",
             "## Quota (achieved vs target)", "",
             "| domain | target | total | real | synthetic | counted | achieved |",
             "|---|--:|--:|--:|--:|--:|:--:|"]
    for dom, q in report["quota"]["by_domain"].items():
        lines.append(f"| {dom} | {q['target']} | {q['total']} | {q['real']} | {q['synthetic']} | "
                     f"{q['counted_toward_target']} | {'✅' if q['achieved'] else '❌'} |")
    lines += ["", "## Dropped by reason", ""]
    lines += [f"- {k}: {v}" for k, v in report["dropped_by_reason"].items()]
    if report["blocked_sources"]:
        lines += ["", "## Blocked sources"]
        lines += [f"- {b['source_id']} ({b['domain']}): {b['reason']}" for b in report["blocked_sources"]]
    if report["quota"]["missed"]:
        lines += ["", "## MISSED quotas"]
        lines += [f"- ❌ {m['domain']}: counted {m['counted']} < target {m['target']}"
                  f"{' (REAL domain)' if m['real_domain'] else ''}" for m in report["quota"]["missed"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(ROOT / "configs" / "data_sources_v3.json"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiments" / "v3_real_domain_generalization.json"))
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw" / "v3"))
    ap.add_argument("--output", default=str(ROOT / "outputs" / "v3-real-domain" / "candidates_v3.jsonl"))
    ap.add_argument("--target-count", type=int, default=100000)
    ap.add_argument("--leakage-index", default=None)
    ap.add_argument("--pii-scan", action="store_true", default=True)
    ap.add_argument("--no-pii-scan", dest="pii_scan", action="store_false")
    ap.add_argument("--allow-no-leakage-index", action="store_true",
                    help="permit building WITHOUT a leakage index — dry-run only")
    ap.add_argument("--fail-on-unknown-license", action="store_true")
    ap.add_argument("--fail-on-domain-quota-miss", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        entries = dsa.load_v3_manifest(args.manifest)        # fail-closed manifest validation
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    cfg = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    fractions = cfg.get("domain_targets") or {}
    targets = {dom: int(round(float(frac) * args.target_count)) for dom, frac in fractions.items()}

    # safety: a full leakage index is REQUIRED to materialize candidates. A real build without
    # one is a hard error; --dry-run (which writes nothing) may proceed without it, with a loud
    # warning (or an explicit --allow-no-leakage-index acknowledgement).
    leakage = None
    if args.leakage_index and pathlib.Path(args.leakage_index).exists():
        leakage = li.LeakageIndex.from_dict(json.loads(pathlib.Path(args.leakage_index).read_text("utf-8")))
    elif not args.dry_run:
        print("ERROR: --leakage-index required to materialize v3 candidates (full eval leakage "
              "scan). It may only be skipped in --dry-run.", file=sys.stderr)
        return 2
    else:
        ack = " (--allow-no-leakage-index)" if args.allow_no_leakage_index else ""
        print(f"WARNING: no leakage index — leakage filtering NOT applied (dry-run only){ack}.",
              file=sys.stderr)

    result = build(entries, args.raw_dir, targets, leakage_index=leakage, pii_scan=args.pii_scan,
                   seed=args.seed)
    report, dropped, quota = result["report"], result["report"]["dropped_by_reason"], result["report"]["quota"]

    status = "ok"
    if args.fail_on_unknown_license and dropped["unknown_license"] > 0:
        status = "fail"
    if args.fail_on_domain_quota_miss and quota["missed"]:
        status = "fail"
    report["status"] = status
    report["leakage_index_used"] = leakage is not None

    out = pathlib.Path(args.output)
    rep_json = out.with_name("candidate_build_v3.json")
    rep_md = out.with_name("candidate_build_v3.md")
    if not args.dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        dp.write_jsonl(str(out), result["candidates"])
        if result["passages"]:
            dp.write_jsonl(str(out.with_name("passages_v3.jsonl")), result["passages"])
        rep_json.parent.mkdir(parents=True, exist_ok=True)
        rep_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        rep_md.write_text(render_markdown(report, status), encoding="utf-8")

    print(json.dumps({"status": status, "totals": report["totals"],
                      "dropped_by_reason": dropped, "missed_quotas": quota["missed"],
                      "leakage_index_used": leakage is not None}, ensure_ascii=False, indent=2))
    print(f"[v3-candidates] status={status} selected={report['totals']['selected_pairs']} "
          f"passages={report['totals']['passages']} missed={len(quota['missed'])} "
          f"dry_run={args.dry_run}", file=sys.stderr)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
