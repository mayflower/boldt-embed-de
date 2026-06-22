#!/usr/bin/env python3
"""Append one auditable row per scored AutoResearch trial to ``outputs/autoresearch/results.tsv``.

Stable columns; header written only when the file does not yet exist; old rows are never rewritten.
Reads ``metrics.json`` (required) and ``score.json`` (optional) from a run directory.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "outputs" / "autoresearch" / "results.tsv"

COLUMNS = [
    "timestamp_utc", "commit", "run_id", "mode", "status", "score",
    "webfaq_recall100", "webfaq_ndcg10", "webfaq_mrr10", "local_rag_recall100",
    "germanquad_ndcg10", "dt_test_ndcg10", "m256_retention", "leakage_hits", "leakage_status",
    "budget_minutes", "elapsed_seconds", "invalid_for_default_loop",
    "vram_gb", "throughput_pairs_per_sec", "config_path", "notes",
]


def _dig(d: Any, *path: str) -> Any:
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _clean(v: Any) -> str:
    if v is None:
        return ""
    return str(v).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def build_row(metrics_doc: Dict[str, Any], score_doc: Optional[Dict[str, Any]],
              status_override: Optional[str], notes: Optional[str],
              timestamp_utc: Optional[str] = None) -> Dict[str, str]:
    """Assemble a results row from a run's metrics.json and optional score.json. Pure function."""
    m = metrics_doc.get("metrics", {})
    status = status_override or (score_doc or {}).get("status") or metrics_doc.get("status")
    score = (score_doc or {}).get("score")
    row = {
        "timestamp_utc": timestamp_utc or "",
        "commit": _dig(metrics_doc, "git", "commit"),
        "run_id": metrics_doc.get("run_id"),
        "mode": metrics_doc.get("mode"),
        "status": status,
        "score": score,
        "webfaq_recall100": _dig(m, "webfaq", "recall@100"),
        "webfaq_ndcg10": _dig(m, "webfaq", "ndcg@10"),
        "webfaq_mrr10": _dig(m, "webfaq", "mrr@10"),
        "local_rag_recall100": _dig(m, "local_rag", "recall@100"),
        "germanquad_ndcg10": _dig(m, "germanquad", "ndcg@10"),
        "dt_test_ndcg10": _dig(m, "dt_test", "ndcg@10"),
        "m256_retention": _dig(m, "matryoshka", "retention_256"),
        "leakage_hits": _dig(m, "leakage", "hits"),
        "leakage_status": _dig(m, "leakage", "status"),
        "budget_minutes": metrics_doc.get("budget_minutes"),
        "elapsed_seconds": metrics_doc.get("elapsed_seconds"),
        "invalid_for_default_loop": metrics_doc.get("invalid_for_default_loop"),
        "vram_gb": _dig(m, "system", "vram_gb"),
        "throughput_pairs_per_sec": _dig(m, "system", "throughput_pairs_per_sec"),
        "config_path": metrics_doc.get("config_path"),
        "notes": notes,
    }
    return {k: _clean(row.get(k)) for k in COLUMNS}


def append_row(results_path: pathlib.Path, row: Dict[str, str]) -> None:
    """Append a row (writing the header first iff the file is new). Never rewrites old rows."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not results_path.exists()
    with open(results_path, "a", encoding="utf-8") as fh:
        if new_file:
            fh.write("\t".join(COLUMNS) + "\n")
        fh.write("\t".join(row.get(c, "") for c in COLUMNS) + "\n")


def _utc_now() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory with metrics.json (+ score.json)")
    ap.add_argument("--results", default=str(DEFAULT_RESULTS))
    ap.add_argument("--status", default=None,
                    help="override: keep|discard|crash|invalid_leakage|invalid_guardrail|"
                         "invalid_for_promotion")
    ap.add_argument("--notes", default=None)
    args = ap.parse_args(argv)

    run_dir = pathlib.Path(args.run)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"error: {metrics_path} not found", flush=True)
        return 2
    metrics_doc = json.loads(metrics_path.read_text(encoding="utf-8"))
    score_path = run_dir / "score.json"
    score_doc = json.loads(score_path.read_text(encoding="utf-8")) if score_path.exists() else None

    row = build_row(metrics_doc, score_doc, args.status, args.notes, timestamp_utc=_utc_now())
    append_row(pathlib.Path(args.results), row)
    print(json.dumps({"appended": True, "run_id": row["run_id"], "status": row["status"],
                      "score": row["score"], "results": args.results}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
