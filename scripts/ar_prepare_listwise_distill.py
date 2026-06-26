#!/usr/bin/env python3
"""Prepare + validate listwise-KL distillation candidate lists (stdlib planner; teacher = flag-gated).

A listwise-KL trial needs candidate lists where each query carries a teacher's ranking
(teacher_score / teacher_softmax_target). This script either VALIDATES existing lists (fail-closed)
or PLANS new teacher-scoring jobs (build_v6_candidate_union -> score_rag_candidate_lists). It never
fabricates teacher scores and never runs Qwen3 unless ``--real --allow-gpu --allow-teacher``.

    python scripts/ar_prepare_listwise_distill.py --config configs/autoresearch/distill/listwise_kl_v8.json --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]


def file_sha256_head(path: Path, n: int = 1 << 20) -> str:
    h = hashlib.sha256(str(path.stat().st_size).encode())
    with path.open("rb") as fh:
        h.update(fh.read(n))
    return h.hexdigest()


def validate_listwise_file(path: Path, *, sample: int = 200) -> Tuple[Dict[str, Any], List[str]]:
    """Validate a listwise candidate-list JSONL. Returns (stats, errors). Fail-closed: a missing
    file, lists with <2 candidates, no positive label, or no teacher signal are errors."""
    errors: List[str] = []
    path = Path(path)
    if not path.exists():
        return {"path": str(path)}, [f"lists file not found: {path}"]
    # eval-leak guard: never distill on an eval corpus
    if "/eval/" in str(path).replace("\\", "/"):
        errors.append(f"lists path looks eval-derived (contains /eval/): {path}")

    n = 0
    with_teacher = 0
    with_positive = 0
    too_few = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n += 1
            if n > sample:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"list #{n} is not valid JSON")
                continue
            if not (rec.get("query") or "").strip():
                errors.append(f"list #{n} has no query")
            cands = rec.get("candidates") or rec.get("docs") or []
            if len(cands) < 2:
                too_few += 1
                continue
            pos = bool(rec.get("positive_doc_ids")) or any(
                (c.get("label") or c.get("high_precision_positive")) for c in cands if isinstance(c, dict))
            if pos:
                with_positive += 1
            teach = any(
                (c.get("teacher_softmax_target") is not None or c.get("teacher_score") is not None)
                for c in cands if isinstance(c, dict))
            if teach:
                with_teacher += 1
    if n == 0:
        errors.append("lists file is empty")
    if too_few:
        errors.append(f"{too_few}/{min(n, sample)} sampled lists have < 2 candidates")
    if n and with_positive == 0:
        errors.append("no sampled list has a positive label (positive_doc_ids / candidate label)")
    if n and with_teacher == 0:
        errors.append("no sampled list carries a teacher signal (teacher_softmax_target/teacher_score)")
    stats = {"path": str(path), "sampled": min(n, sample), "with_teacher": with_teacher,
             "with_positive": with_positive, "lists_too_few_candidates": too_few,
             "list_hash": file_sha256_head(path) if path.exists() else None}
    return stats, errors


def plan_new_teacher_lists(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build the (planned) command sequence to score NEW lists with the teacher. Plan only."""
    nt = cfg.get("new_teacher_lists", {}) or {}
    sources = nt.get("source_ids", [])
    slice_rows = int(nt.get("slice_rows", 50000))
    pools = ",".join(nt.get("candidate_pools", ["bm25", "dense_current"]))
    build = ["python", "scripts/build_v6_candidate_union.py",
             "--sources", ",".join(sources), "--slice-rows", str(slice_rows),
             "--candidate-pools", pools, "--out", "outputs/autoresearch/distill/new_lists/union.jsonl"]
    score = ["python", "scripts/score_rag_candidate_lists.py",
             "--lists", "outputs/autoresearch/distill/new_lists/union.jsonl",
             "--teacher-config", "configs/teacher_models.json",
             "--out", "outputs/autoresearch/distill/new_lists/teacher_scored.jsonl"]
    return {"enabled": True, "teacher": nt.get("teacher", "qwen3-reranker-8b"),
            "source_ids": sources, "slice_rows": slice_rows,
            "planned_commands": [" ".join(build), " ".join(score)],
            "note": "GPU-days teacher inference; runs only with --real --allow-gpu --allow-teacher; "
                    "de-risk with slice_rows before the full set"}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="outputs/autoresearch/distill")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--allow-gpu", action="store_true")
    ap.add_argument("--allow-teacher", action="store_true")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    out_dir = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    nt = cfg.get("new_teacher_lists", {}) or {}
    manifest: Dict[str, Any] = {"name": cfg.get("name"), "lists": cfg.get("lists")}

    if nt.get("enabled"):
        plan = plan_new_teacher_lists(cfg)
        manifest["new_teacher_lists"] = plan
        (out_dir / "prepare_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if not (args.real and args.allow_gpu and args.allow_teacher):
            print(json.dumps({"status": "planned", **plan}, ensure_ascii=False, indent=2))
            return 0
        for cmd in plan["planned_commands"]:
            proc = subprocess.run([sys.executable] + cmd.split()[1:], cwd=str(ROOT))
            if proc.returncode != 0:
                print(json.dumps({"status": "fail", "failed_command": cmd}, ensure_ascii=False))
                return proc.returncode
        print(json.dumps({"status": "ok", "note": "teacher lists scored"}, ensure_ascii=False))
        return 0

    # existing-lists path: validate fail-closed
    lists = cfg.get("lists")
    if not lists:
        msg = "config has no 'lists' path and new_teacher_lists is not enabled"
        (out_dir / "prepare_manifest.json").write_text(
            json.dumps({**manifest, "errors": [msg]}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "fail", "errors": [msg]}, ensure_ascii=False, indent=2))
        return 1
    stats, errors = validate_listwise_file(ROOT / lists if not Path(lists).is_absolute()
                                           else Path(lists))
    manifest["validation"] = stats
    manifest["errors"] = errors
    (out_dir / "prepare_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "fail" if errors else "ok", "validation": stats, "errors": errors},
                     ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
