#!/usr/bin/env python3
"""AutoResearch data preparation — build stdlib-safe manifests from LOCAL files only.

NO downloads, NO Hugging Face fetches. Operates only on files supplied by the user or produced by
existing repository pipelines. Writes ``prepare_manifest.json``, ``train_summary.json``, and
``eval_summary.json`` so a trial is reproducible and auditable.

A preparation is **promotable only** when a leakage report with zero hits is supplied
(``--require-leakage-report``). Without one, the manifest is marked ``leakage_status:
"not_checked"`` and ``promotable: false``.
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

from boldt_embed import experiment_registry as registry  # noqa: E402
from boldt_embed import leakage_index  # noqa: E402

REQUIRED_TRAIN_FIELDS = ("query_id", "query", "document")


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _git_commit() -> Optional[str]:
    commit = registry.current_git_commit()  # reuse the repo's canonical capture
    return None if commit in ("", "unknown") else commit


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_info(path: pathlib.Path) -> Dict[str, Any]:
    p = pathlib.Path(path)
    info: Dict[str, Any] = {"path": str(p), "exists": p.exists()}
    try:
        info["repo_relative"] = str(p.resolve().relative_to(ROOT))
    except ValueError:
        info["repo_relative"] = None
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


def _count_field(records: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in records:
        v = r.get(field)
        if v is None:
            continue
        counts[str(v)] = counts.get(str(v), 0) + 1
    return counts


def summarize_train(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts, per-field missing-required counts, and source/domain/license breakdowns."""
    missing_required = {f: 0 for f in REQUIRED_TRAIN_FIELDS}
    records_missing_any = 0
    for r in records:
        miss = [f for f in REQUIRED_TRAIN_FIELDS
                if r.get(f) in (None, "") or (isinstance(r.get(f), str) and not r[f].strip())]
        for f in miss:
            missing_required[f] += 1
        if miss:
            records_missing_any += 1
    return {
        "count": len(records),
        "required_fields": list(REQUIRED_TRAIN_FIELDS),
        "missing_required": missing_required,
        "records_missing_any_required": records_missing_any,
        "source_counts": _count_field(records, "source"),
        "domain_counts": _count_field(records, "domain"),
        "license_counts": _count_field(records, "license"),
    }


# Fields the repo's real leakage scan (leakage_index / run_full_leakage_scan.py) writes.
_LEAKAGE_INDEX_FIELDS = ("exact_hits", "exact_normalized_hits", "near_duplicate_hits")
_LEAKAGE_RECOGNIZED = ("hits", "num_hits", "leakage_hits") + _LEAKAGE_INDEX_FIELDS


def _num(v: Any) -> Optional[int]:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def looks_like_leakage_report(report: Dict[str, Any]) -> bool:
    """True if the JSON has any recognized leakage field (so an empty/unrelated file isn't
    silently treated as a clean report)."""
    return (any(k in report for k in _LEAKAGE_RECOGNIZED)
            or isinstance(report.get("summary"), dict))


def extract_leakage_hits(report: Dict[str, Any]) -> Optional[int]:
    """Pull a hit count, tolerating both the simple {hits: N} shape and the repo's real
    leakage_index report (exact_hits + exact_normalized_hits + near_duplicate_hits). None if
    no recognized field is present."""
    for key in ("hits", "num_hits", "leakage_hits"):
        n = _num(report.get(key))
        if n is not None:
            return n
    li = [_num(report.get(k)) for k in _LEAKAGE_INDEX_FIELDS]
    if any(x is not None for x in li):
        return sum(x for x in li if x is not None)
    summ = report.get("summary")
    if isinstance(summ, dict) and _num(summ.get("hits")) is not None:
        return _num(summ["hits"])
    return None


def evaluate_eval_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve each eval set's file, classifying required-missing (fatal) vs optional-missing."""
    sets_out: List[Dict[str, Any]] = []
    missing_required: List[str] = []
    missing_optional: List[str] = []
    for s in manifest.get("sets", []):
        name = s.get("name")
        path = s.get("path")
        optional = bool(s.get("optional", False)) or s.get("role") == "primary_optional"
        exists = bool(path) and (ROOT / path).exists() if path else False
        entry = {"name": name, "role": s.get("role"), "path": path,
                 "optional": optional, "exists": exists}
        if path:
            entry.update({k: v for k, v in file_info(ROOT / path).items()
                          if k in ("size_bytes", "sha256", "repo_relative")})
            if exists:
                try:
                    entry["record_count"] = len(read_jsonl(ROOT / path))
                except Exception as exc:
                    entry["record_count_error"] = str(exc)
        if not exists:
            (missing_optional if optional else missing_required).append(name or path or "?")
        sets_out.append(entry)
    return {"sets": sets_out, "missing_required": missing_required,
            "missing_optional": missing_optional}


def build_manifest(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any],
                                                      Dict[str, Any], int]:
    """Return (prepare_manifest, train_summary, eval_summary, exit_code)."""
    train_path = pathlib.Path(args.train)
    train_records = read_jsonl(train_path, args.max_records) if train_path.exists() else []
    train_summary = summarize_train(train_records)
    train_summary["file"] = file_info(train_path)
    train_summary["max_records_cap"] = args.max_records

    eval_manifest = json.loads(pathlib.Path(args.eval_manifest).read_text(encoding="utf-8"))
    eval_summary = evaluate_eval_manifest(eval_manifest)

    # --- leakage ---
    leakage: Dict[str, Any] = {"status": "not_checked", "report": None, "hits": None}
    if args.require_leakage_report:
        rep_path = pathlib.Path(args.require_leakage_report)
        if not rep_path.exists():
            leakage["status"] = "missing_report"
        else:
            report = json.loads(rep_path.read_text(encoding="utf-8"))
            hits = extract_leakage_hits(report)
            leakage["hits"] = hits
            leakage["report"] = {"path": str(rep_path), "summary": report.get("summary", report)}
            if not looks_like_leakage_report(report):
                leakage["status"] = "unparseable"
            else:
                if any(k in report for k in _LEAKAGE_INDEX_FIELDS):
                    # reuse the repo's canonical decision for its own report format
                    # (honors the cleaned_candidates_path escape hatch)
                    clean = leakage_index.leakage_report_is_clean(report)
                else:
                    clean = (hits == 0) or bool(report.get("cleaned_candidates_path"))
                leakage["status"] = "clean" if clean else "leak_detected"

    exit_code = 0
    fatal: List[str] = []
    if not train_path.exists():
        fatal.append(f"train file missing: {train_path}")
    if eval_summary["missing_required"]:
        fatal.append(f"required eval set(s) missing: {eval_summary['missing_required']}")
    if args.require_leakage_report and leakage["status"] != "clean":
        fatal.append(f"leakage gate not clean: status={leakage['status']} hits={leakage['hits']}")
    if fatal:
        exit_code = 1

    promotable = (not fatal) and leakage["status"] == "clean"
    manifest = {
        "timestamp_utc": _utc_now(),
        "git_commit": _git_commit(),
        "seed": args.seed,
        "baseline_model": args.baseline_model,
        "train": train_summary,
        "eval": eval_summary,
        "leakage": leakage,
        "promotable": promotable,
        "fatal": fatal,
        "note": ("Promotable preparation." if promotable else
                 "NOT promotable — supply a clean leakage report and all required eval sets."),
    }
    return manifest, train_summary, eval_summary, exit_code


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True, help="JSONL candidate records")
    ap.add_argument("--eval-manifest", required=True, help="JSON manifest listing eval sets")
    ap.add_argument("--baseline-model", required=True, help="reference model id or local path")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--max-records", type=int, default=None, help="cap for proxy preparation")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--require-leakage-report", default=None,
                    help="leakage report; preparation fails (and is not promotable) if hits > 0")
    ap.add_argument("--format", choices=["json", "markdown"], default="json")
    args = ap.parse_args(argv)

    manifest, train_summary, eval_summary, exit_code = build_manifest(args)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prepare_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "train_summary.json").write_text(
        json.dumps(train_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "eval_summary.json").write_text(
        json.dumps(eval_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.format == "markdown":
        print(f"# AutoResearch preparation\n\n- promotable: **{manifest['promotable']}**\n"
              f"- leakage: {manifest['leakage']['status']} (hits={manifest['leakage']['hits']})\n"
              f"- train records: {train_summary['count']}\n"
              f"- eval sets: {len(eval_summary['sets'])} "
              f"(missing required: {eval_summary['missing_required']})")
    else:
        print(json.dumps({"promotable": manifest["promotable"],
                          "leakage_status": manifest["leakage"]["status"],
                          "train_records": train_summary["count"],
                          "missing_required_eval": eval_summary["missing_required"],
                          "fatal": manifest["fatal"]}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
