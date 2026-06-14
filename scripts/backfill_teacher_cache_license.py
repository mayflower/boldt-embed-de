#!/usr/bin/env python3
"""Back-fill license provenance into a teacher cache that was built before the provenance fix.

Joins cache rows to their candidate set on (query_id, doc_id), re-derives full provenance from
the source manifest (`source_manifest.candidate_provenance`), rewrites the cache shards in
place, and regenerates the summary. Pure stdlib, no ML. One-off remediation for the v2 cache
whose summary reported `by_license {"unknown": N}`.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import source_manifest as sm  # noqa: E402
from boldt_embed import teacher as T  # noqa: E402


def _provenance_map(candidates_path, manifest_path):
    entries = {e.source_id: e for e in sm.load_source_manifest(manifest_path)}
    m = {}
    for line in pathlib.Path(candidates_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        sid = str(c.get("source") or (c.get("metadata") or {}).get("source_id") or "")
        entry = entries.get(sid)
        if entry is None:
            continue
        prov = sm.candidate_provenance(entry, c)
        prov["domain"] = str(c.get("domain") or entry.domain)
        m[(c["query_id"], c["doc_id"])] = prov
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(ROOT / "configs" / "data_sources_v2.json"))
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--cache-manifest", required=True, help="<prefix>.manifest.json of the cache")
    ap.add_argument("--summary-output", required=True)
    args = ap.parse_args()

    prov = _provenance_map(args.candidates, args.manifest)
    man = json.loads(pathlib.Path(args.cache_manifest).read_text(encoding="utf-8"))
    enriched_total = matched = 0
    all_rows = []
    for sh in man.get("shards", []):
        sp = pathlib.Path(sh["path"])
        if not sp.exists():
            continue
        rows = T.read_teacher_cache_jsonl(sp)
        for r in rows:
            p = prov.get((r.get("query_id"), r.get("doc_id")))
            if p:
                matched += 1
                for k, v in p.items():
                    if r.get(k) in (None, "unknown"):
                        r[k] = v
            enriched_total += 1
        T.write_teacher_cache_jsonl(sp, rows)  # rewrite shard in place (gitignored)
        all_rows += rows
    summary = T.summarize_cache(all_rows)
    pathlib.Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    print(f"[backfill] rows={enriched_total} matched={matched} "
          f"unknown_license_rows={summary['unknown_license_rows']} -> {args.summary_output}")
    print(f"[backfill] by_license={json.dumps(summary['by_license'], ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
