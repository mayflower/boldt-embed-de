#!/usr/bin/env python3
"""AutoResearch — on-policy hard-negative refresh (pure stdlib, fail-closed).

Refreshes hard negatives for the *current* training mixture by mining multiple candidate
pools, dropping likely false negatives with a teacher-score-gated filter, balancing domains,
and emitting BOTH a hard-negative/triplet set and per-query listwise reranker candidates.

Everything here runs on the Python standard library. The BM25-only path actually mines; the
``dense_current`` / ``dense_teacher`` pools consume *precomputed* embedding artifacts and
**fail closed with a clear missing-input message** when those artifacts are not supplied.
We NEVER run a model (no Qwen3, no GPU) and NEVER fabricate teacher scores — teacher scores
are loaded from a provided cache file via ``negative_mining_2026.load_teacher_scores``.

Outputs (under ``--out``):
  - hardnegatives.jsonl        one row per query (build_triplets_or_lists schema)
  - listwise_candidates.jsonl  one row per query (build_reranker_candidate_lists schema)
  - manifest.json              pools, filter stats, kept/dropped, per-domain balance, hashes
  - report.md                  human-readable summary

The ``false_negative_filter`` supports ``method:"margin_or_ratio"``: a candidate is treated
as a likely false negative (dropped) when its teacher score is within ``margin`` of the
positive OR is at least ``ratio`` * positive. Both are expressed as a per-query *effective
margin* ``max(margin, pos*(1-ratio))`` and handed to the reused module's margin logic, so the
drop rule is not duplicated here.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import negative_mining_2026 as nm  # noqa: E402

# Pools that can run with no extra artifacts vs. pools that REQUIRE precomputed embeddings.
STDLIB_POOLS = {"bm25"}
DENSE_POOLS = {"dense_current", "dense_teacher"}
# config key holding the precomputed-embeddings artifact path for each dense pool.
DENSE_ARTIFACT_KEY = {
    "dense_current": "dense_current_embeddings",
    "dense_teacher": "dense_teacher_embeddings",
}


class FailClosed(Exception):
    """Raised for a missing/invalid input; surfaced as a clear message and a nonzero exit."""


# --------------------------------------------------------------------------- io helpers
def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_info(path: pathlib.Path) -> Dict[str, Any]:
    p = pathlib.Path(path)
    info: Dict[str, Any] = {"path": str(p), "exists": p.exists()}
    if p.exists() and p.is_file():
        info["abspath"] = str(p.resolve())
        info["size_bytes"] = p.stat().st_size
        info["sha256"] = sha256_file(p)
    return info


def read_jsonl(path: pathlib.Path, max_records: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_records is not None and len(rows) >= max_records:
                break
    return rows


def write_jsonl(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------------- config parse
def load_config(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        raise FailClosed(f"config file missing: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg.setdefault("candidate_pools", ["bm25"])
    cfg.setdefault("negatives_per_query", 8)
    cfg.setdefault("candidate_list_k", 32)
    cfg.setdefault("false_negative_filter", {})
    cfg.setdefault("domain_balance", {})
    return cfg


def filter_params(cfg: Dict[str, Any]) -> Tuple[str, float, Optional[float]]:
    """(method, margin, ratio). Defaults: margin_or_ratio, margin 0.1, no ratio."""
    f = cfg.get("false_negative_filter") or {}
    method = str(f.get("method", "margin_or_ratio"))
    margin = float(f.get("margin", 0.1))
    ratio = f.get("ratio", None)
    ratio = float(ratio) if ratio is not None else None
    return method, margin, ratio


def effective_margin(pos_score: Optional[float], method: str, margin: float,
                     ratio: Optional[float]) -> float:
    """Convert the configured rule into a per-query margin handed to the reused module.

    ``margin`` rule:           drop if neg >= pos - margin.
    ``margin_or_ratio`` rule:  drop if (neg >= pos - margin) OR (neg >= ratio*pos), which is
                               the single margin ``max(margin, pos*(1-ratio))``."""
    if method == "margin_or_ratio" and ratio is not None and pos_score is not None:
        return max(margin, pos_score * (1.0 - ratio))
    return margin


# ----------------------------------------------------------------------------- pools
def resolve_pools(cfg: Dict[str, Any]) -> Tuple[List[str], List[Tuple[str, str, pathlib.Path]]]:
    """Validate requested pools. Returns (stdlib_pool_names, dense_pool_specs).

    Each dense spec is ``(pool_name, artifact_key, artifact_path)``. Fails closed for an
    unknown pool or a dense pool whose embeddings artifact is missing/unset."""
    requested = list(cfg.get("candidate_pools", []))
    if not requested:
        raise FailClosed("candidate_pools is empty — nothing to mine.")
    stdlib_names: List[str] = []
    dense_specs: List[Tuple[str, str, pathlib.Path]] = []
    for pool in requested:
        if pool in STDLIB_POOLS:
            stdlib_names.append(pool)
        elif pool in DENSE_POOLS:
            key = DENSE_ARTIFACT_KEY[pool]
            art = cfg.get(key)
            if not art:
                raise FailClosed(
                    f"pool '{pool}' requires precomputed embeddings, but config key '{key}' "
                    f"is unset. Provide a JSON artifact with query/doc embeddings, or remove "
                    f"'{pool}' from candidate_pools. (No model is run here — fail closed.)")
            art_path = pathlib.Path(art)
            if not art_path.is_absolute():
                art_path = ROOT / art_path
            if not art_path.exists():
                raise FailClosed(
                    f"pool '{pool}' embeddings artifact not found: {art_path} (config key "
                    f"'{key}'). Precompute embeddings first — fail closed, no model is loaded.")
            dense_specs.append((pool, key, art_path))
        else:
            raise FailClosed(
                f"unknown candidate pool '{pool}'. Supported: "
                f"{sorted(STDLIB_POOLS | DENSE_POOLS)}.")
    return stdlib_names, dense_specs


# -------------------------------------------------------------------------- core build
def to_positives_corpus(records: List[Dict[str, Any]]
                        ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]],
                                   Dict[str, Dict[str, Any]]]:
    """Map mixture rows {query_id,query,document[,source,domain]} into the module's shapes.

    Each row is a (query, positive) pair; the positive document is also a corpus doc keyed by
    its ``doc_id`` (using the row's ``doc_id`` if present, else the ``query_id``). Returns
    (positives, corpus_rows_for_bm25, corpus_lookup)."""
    positives: List[Dict[str, Any]] = []
    corpus_rows: List[Dict[str, Any]] = []
    corpus_lookup: Dict[str, Dict[str, Any]] = {}
    for i, r in enumerate(records):
        qid = str(r.get("query_id", i))
        doc_id = str(r.get("doc_id", r.get("positive_doc_id", qid)))
        text = r.get("document") or r.get("positive") or r.get("text") or ""
        domain = str(r.get("domain", "unknown"))
        source = str(r.get("source", "unknown"))
        positives.append({"query_id": qid, "query": r.get("query", ""), "doc_id": doc_id,
                          "document": text, "domain": domain, "source": source})
        if doc_id not in corpus_lookup:
            doc = {"doc_id": doc_id, "text": text, "domain": domain, "source": source}
            corpus_lookup[doc_id] = doc
            corpus_rows.append(doc)
    return positives, corpus_rows, corpus_lookup


def load_dense_pool(pool: str, art_path: pathlib.Path, corpus_lookup: Dict[str, Dict[str, Any]],
                    k: int) -> Dict[str, List[str]]:
    """Mine top-k doc ids from a precomputed-embeddings artifact (no model loaded).

    Artifact JSON shape:
      {"query_embeddings": {qid: [floats...]},
       "doc_embeddings": [[doc_id, [floats...]], ...]}"""
    art = json.loads(art_path.read_text(encoding="utf-8"))
    q_emb = art.get("query_embeddings")
    d_emb = art.get("doc_embeddings")
    if not isinstance(q_emb, dict) or not isinstance(d_emb, list):
        raise FailClosed(
            f"pool '{pool}' artifact {art_path} is malformed: expected keys "
            f"'query_embeddings' (obj) and 'doc_embeddings' (list of [doc_id, vector]).")
    doc_embeddings = [(str(d[0]), d[1]) for d in d_emb]
    return nm.mine_dense_candidates_from_embeddings(
        {str(qid): vec for qid, vec in q_emb.items()}, doc_embeddings, k=k)


def build_outputs(cfg: Dict[str, Any], records: List[Dict[str, Any]],
                  teacher_rows: Optional[List[Dict[str, Any]]],
                  stdlib_names: List[str],
                  dense_specs: List[Tuple[str, str, pathlib.Path]]
                  ) -> Tuple[List[Dict[str, Any]], Dict[str, Any],
                             List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """Run the real mining pipeline. Returns
    (hardneg_rows, hardneg_stats, listwise_rows, listwise_stats, pool_descriptors)."""
    positives, corpus_rows, corpus_lookup = to_positives_corpus(records)
    queries = [{"query_id": p["query_id"], "query": p["query"]} for p in positives]
    k = int(cfg.get("candidate_list_k", 32))

    pools: List[Tuple[str, Dict[str, List[str]]]] = []
    pool_descriptors: List[Dict[str, Any]] = []
    for pool in stdlib_names:  # currently just bm25
        cand = nm.mine_bm25_candidates(queries, corpus_rows, k=k)
        pools.append((pool, cand))
        pool_descriptors.append({"name": pool, "type": "stdlib_bm25", "top_k": k,
                                 "queries_with_candidates": sum(1 for v in cand.values() if v)})
    for pool, key, art_path in dense_specs:
        cand = load_dense_pool(pool, art_path, corpus_lookup, k)
        pools.append((pool, cand))
        pool_descriptors.append({"name": pool, "type": "precomputed_embeddings",
                                 "artifact": str(art_path), "config_key": key, "top_k": k,
                                 "queries_with_candidates": sum(1 for v in cand.values() if v)})

    merged = nm.merge_candidate_pools(*pools)
    teacher_scores = nm.load_teacher_scores(teacher_rows or [])

    method, margin, ratio = filter_params(cfg)
    max_per_domain = (cfg.get("domain_balance") or {}).get("max_per_domain")
    n_per_q = int(cfg.get("negatives_per_query", 8))

    # Per-query effective margin folds the ratio rule into the reused module's margin logic.
    # When method=='margin' or no teacher scores, this equals the configured margin.
    score_for = nm._filter_score
    hardneg_rows: List[Dict[str, Any]] = []
    listwise_rows: List[Dict[str, Any]] = []
    hn_acc = _StatAccum()
    lw_acc = _StatAccum(listwise=True)
    for pos in positives:
        pos_score = score_for(teacher_scores.get((pos["query_id"], pos["doc_id"])))
        eff = effective_margin(pos_score, method, margin, ratio)
        hn, hn_s = nm.build_triplets_or_lists(
            [pos], merged, corpus_lookup, teacher_scores,
            negatives_per_query=n_per_q, margin=eff, max_per_domain=max_per_domain)
        lw, lw_s = nm.build_reranker_candidate_lists(
            [pos], merged, corpus_lookup, teacher_scores,
            negatives_per_query=n_per_q, margin=eff)
        hardneg_rows.extend(hn)
        listwise_rows.extend(lw)
        hn_acc.add(hn_s)
        lw_acc.add(lw_s)
    return hardneg_rows, hn_acc.finalize(), listwise_rows, lw_acc.finalize(), pool_descriptors


class _StatAccum:
    """Merge the per-query stats dicts returned by the module into one aggregate."""

    def __init__(self, listwise: bool = False):
        self.listwise = listwise
        self.int_keys = (["queries", "positives", "negatives", "vetoed_false_negatives"]
                         if listwise else ["queries", "total_candidates", "kept"])
        self.dict_keys = (["candidates_by_source", "candidates_by_domain"] if listwise
                          else ["dropped_by_reason", "kept_by_source", "kept_by_domain"])
        self.ints: Dict[str, int] = {k: 0 for k in self.int_keys}
        self.dicts: Dict[str, Dict[str, int]] = {k: {} for k in self.dict_keys}

    def add(self, s: Dict[str, Any]) -> None:
        for k in self.int_keys:
            self.ints[k] += int(s.get(k, 0))
        for k in self.dict_keys:
            for kk, vv in (s.get(k) or {}).items():
                self.dicts[k][kk] = self.dicts[k].get(kk, 0) + vv

    def finalize(self) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(self.ints)
        for k in self.dict_keys:
            out[k] = dict(sorted(self.dicts[k].items()))
        return out


# ------------------------------------------------------------------------ manifest/report
def build_manifest(cfg: Dict[str, Any], args: argparse.Namespace, *, dry_run: bool,
                   queries_info: Dict[str, Any], corpus_info: Dict[str, Any],
                   teacher_info: Optional[Dict[str, Any]], record_count: int,
                   pool_descriptors: List[Dict[str, Any]],
                   hardneg_stats: Optional[Dict[str, Any]],
                   listwise_stats: Optional[Dict[str, Any]],
                   now: str) -> Dict[str, Any]:
    method, margin, ratio = filter_params(cfg)
    max_per_domain = (cfg.get("domain_balance") or {}).get("max_per_domain")
    manifest: Dict[str, Any] = {
        "tool": "ar_refresh_hardnegatives",
        "label": cfg.get("name", pathlib.Path(args.out).name),
        "timestamp_utc": now,
        "dry_run": dry_run,
        "config_path": str(args.config),
        "out_dir": str(args.out),
        "inputs": {
            "queries": queries_info,
            "corpus": corpus_info,
            "teacher_scores": teacher_info,
            "record_count": record_count,
        },
        "candidate_pools": {
            "requested": list(cfg.get("candidate_pools", [])),
            "resolved": pool_descriptors,
        },
        "false_negative_filter": {
            "method": method, "margin": margin, "ratio": ratio,
            "teacher_scores_present": bool(teacher_info and teacher_info.get("exists")),
        },
        "domain_balance": {"max_per_domain": max_per_domain},
        "negatives_per_query": int(cfg.get("negatives_per_query", 8)),
        "candidate_list_k": int(cfg.get("candidate_list_k", 32)),
        "outputs": {
            "hardnegatives": "hardnegatives.jsonl",
            "listwise_candidates": "listwise_candidates.jsonl",
        },
    }
    if dry_run:
        manifest["plan"] = (
            "DRY RUN — validated config, inputs, and pools; no mining performed. "
            "Re-run without --dry-run to mine and write the JSONL outputs.")
    else:
        kept = hardneg_stats.get("kept", 0)
        dropped = sum((hardneg_stats.get("dropped_by_reason") or {}).values())
        manifest["filter_statistics"] = {
            "total_candidates": hardneg_stats.get("total_candidates", 0),
            "kept": kept,
            "dropped": dropped,
            "dropped_by_reason": hardneg_stats.get("dropped_by_reason", {}),
            "listwise_vetoed_false_negatives":
                listwise_stats.get("vetoed_false_negatives", 0),
        }
        manifest["per_domain_balance"] = {
            "hardneg_kept_by_domain": hardneg_stats.get("kept_by_domain", {}),
            "hardneg_kept_by_source": hardneg_stats.get("kept_by_source", {}),
            "listwise_candidates_by_domain": listwise_stats.get("candidates_by_domain", {}),
            "listwise_candidates_by_source": listwise_stats.get("candidates_by_source", {}),
        }
        manifest["hardneg_stats"] = hardneg_stats
        manifest["listwise_stats"] = listwise_stats
    return manifest


def render_report(manifest: Dict[str, Any]) -> str:
    lines = [f"# Hard-negative refresh — {manifest['label']}", ""]
    lines.append(f"- timestamp (UTC): {manifest['timestamp_utc']}")
    lines.append(f"- dry-run: **{manifest['dry_run']}**")
    lines.append(f"- queries file: `{manifest['inputs']['queries'].get('path')}` "
                 f"(records: {manifest['inputs']['record_count']})")
    fnf = manifest["false_negative_filter"]
    lines.append(f"- false-negative filter: {fnf['method']} "
                 f"(margin={fnf['margin']}, ratio={fnf['ratio']}, "
                 f"teacher_scores={fnf['teacher_scores_present']})")
    lines.append(f"- domain balance: max_per_domain="
                 f"{manifest['domain_balance']['max_per_domain']}")
    lines.append("")
    lines.append("## Candidate pools")
    for p in manifest["candidate_pools"]["resolved"]:
        lines.append(f"- **{p['name']}** ({p['type']}, top_k={p.get('top_k')}, "
                     f"queries_with_candidates={p.get('queries_with_candidates', 'n/a')})")
    if manifest["dry_run"]:
        lines += ["", f"_{manifest['plan']}_"]
        return "\n".join(lines) + "\n"
    fs = manifest["filter_statistics"]
    lines += ["", "## Filter statistics",
              f"- total candidates: {fs['total_candidates']}",
              f"- kept: {fs['kept']}",
              f"- dropped (likely false negatives): {fs['dropped']}",
              f"- dropped by reason: {fs['dropped_by_reason']}",
              f"- listwise vetoed false negatives: {fs['listwise_vetoed_false_negatives']}",
              "", "## Per-domain balance (hard negatives kept)",
              f"- by domain: {manifest['per_domain_balance']['hardneg_kept_by_domain']}",
              f"- by source: {manifest['per_domain_balance']['hardneg_kept_by_source']}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------- driver
def run(args: argparse.Namespace, now: Optional[str] = None
        ) -> Tuple[Dict[str, Any], int]:
    """Execute the refresh. Returns (manifest, exit_code). ``now`` is injectable for tests."""
    now = now or _utc_now()
    cfg = load_config(pathlib.Path(args.config))

    q_path = pathlib.Path(cfg.get("queries", ""))
    c_path = pathlib.Path(cfg.get("corpus", ""))
    if not str(cfg.get("queries", "")):
        raise FailClosed("config has no 'queries' file.")
    if not str(cfg.get("corpus", "")):
        raise FailClosed("config has no 'corpus' file.")
    if not q_path.is_absolute():
        q_path = ROOT / q_path
    if not c_path.is_absolute():
        c_path = ROOT / c_path
    if not q_path.exists():
        raise FailClosed(f"queries file missing: {q_path}")
    if not c_path.exists():
        raise FailClosed(f"corpus file missing: {c_path}")

    queries_info = file_info(q_path)
    corpus_info = file_info(c_path)

    # Teacher scores: CLI flag wins; else config. Loaded only — never fabricated.
    teacher_path = args.teacher_scores or cfg.get("teacher_scores")
    teacher_info: Optional[Dict[str, Any]] = None
    teacher_rows: Optional[List[Dict[str, Any]]] = None
    if teacher_path:
        tp = pathlib.Path(teacher_path)
        if not tp.is_absolute():
            tp = ROOT / tp
        if not tp.exists():
            raise FailClosed(f"teacher-scores file missing: {tp}")
        teacher_info = file_info(tp)
        teacher_rows = read_jsonl(tp)

    # Validate pools (fails closed for dense pools without artifacts) BEFORE any heavy work.
    stdlib_names, dense_specs = resolve_pools(cfg)

    records = read_jsonl(q_path, args.max_records)
    record_count = len(records)

    if args.dry_run:
        # Describe pools without mining.
        pool_descriptors = [{"name": n, "type": "stdlib_bm25",
                             "top_k": int(cfg.get("candidate_list_k", 32))}
                            for n in stdlib_names]
        pool_descriptors += [{"name": n, "type": "precomputed_embeddings",
                              "artifact": str(p), "config_key": k,
                              "top_k": int(cfg.get("candidate_list_k", 32))}
                             for n, k, p in dense_specs]
        manifest = build_manifest(cfg, args, dry_run=True, queries_info=queries_info,
                                  corpus_info=corpus_info, teacher_info=teacher_info,
                                  record_count=record_count, pool_descriptors=pool_descriptors,
                                  hardneg_stats=None, listwise_stats=None, now=now)
        _write_outputs(args, manifest, hardneg_rows=None, listwise_rows=None)
        return manifest, 0

    (hardneg_rows, hardneg_stats, listwise_rows, listwise_stats,
     pool_descriptors) = build_outputs(cfg, records, teacher_rows, stdlib_names, dense_specs)
    manifest = build_manifest(cfg, args, dry_run=False, queries_info=queries_info,
                              corpus_info=corpus_info, teacher_info=teacher_info,
                              record_count=record_count, pool_descriptors=pool_descriptors,
                              hardneg_stats=hardneg_stats, listwise_stats=listwise_stats, now=now)
    _write_outputs(args, manifest, hardneg_rows=hardneg_rows, listwise_rows=listwise_rows)
    return manifest, 0


def _write_outputs(args: argparse.Namespace, manifest: Dict[str, Any],
                   hardneg_rows: Optional[List[Dict[str, Any]]],
                   listwise_rows: Optional[List[Dict[str, Any]]]) -> None:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(render_report(manifest), encoding="utf-8")
    # Always write the JSONL outputs (empty on dry-run) so downstream paths are stable.
    write_jsonl(out / "hardnegatives.jsonl", hardneg_rows or [])
    write_jsonl(out / "listwise_candidates.jsonl", listwise_rows or [])


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="hard-negative refresh JSON config")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--teacher-scores", default=None,
                    help="JSONL teacher-cache rows (query_id, doc_id, embedding_score, "
                         "reranker_score); overrides config.teacher_scores. Never fabricated.")
    ap.add_argument("--max-records", type=int, default=None,
                    help="cap queries processed (proxy refresh)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate config/inputs/pools and write a plan manifest; no mining")
    ap.add_argument("--format", choices=["json", "markdown"], default="json")
    ap.add_argument("--timestamp", default=None,
                    help="override UTC timestamp (determinism for tests/reproduction)")
    args = ap.parse_args(argv)

    try:
        manifest, exit_code = run(args, now=args.timestamp)
    except FailClosed as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
              file=sys.stderr)
        return 2

    if args.format == "markdown":
        print(render_report(manifest))
    else:
        summary = {"ok": True, "dry_run": manifest["dry_run"], "label": manifest["label"],
                   "record_count": manifest["inputs"]["record_count"],
                   "pools": [p["name"] for p in manifest["candidate_pools"]["resolved"]]}
        if not manifest["dry_run"]:
            summary["filter_statistics"] = manifest["filter_statistics"]
        print(json.dumps(summary, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
