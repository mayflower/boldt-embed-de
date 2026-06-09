#!/usr/bin/env python3
"""Evaluate reranker LIFT over FIXED candidate sets (Prompt 7).

Input is a JSONL of fixed shortlists — `{query_id, query, candidates:[{doc_id, document}],
positive_ids:[...]}` — so the reranker is measured exactly where it acts (re-ordering a given
first stage), never confounded with a retriever. Reports nDCG@10 for:

* first-stage (candidates as given),
* first-stage + student reranker,
* first-stage + teacher reranker (optional),
* oracle (positives first) and positive-in-top-k.

First-stage and oracle metrics are stdlib (computed even in `--dry-run`); the student/teacher
rerankers need the `train`/`eval` extras + GPU.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data_pipeline as dp  # noqa: E402
from boldt_embed import experiment_registry as ER  # noqa: E402
from boldt_embed import reranker_modern as RM  # noqa: E402
from boldt_embed.config import load_reranker_config  # noqa: E402


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _positives(row):
    return set(row.get("positive_ids") or row.get("qrels") or [])


def _first_stage_and_oracle(rows, k):
    fs, orc, pintk = [], [], []
    for r in rows:
        cids = [c["doc_id"] for c in r["candidates"]]
        pos = _positives(r)
        fs.append(RM.first_stage_metrics(cids, pos, (k,)))
        orc.append(RM.oracle_metrics(cids, pos, (k,)))
        pintk.append({f"pos_in_top_{k}": RM.positive_in_top_k(
            cids, list(range(len(cids), 0, -1)), pos, k)})
    return fs, orc, pintk


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", required=True, help="fixed-shortlist JSONL")
    ap.add_argument("--config", default=str(ROOT / "configs" / "training_reranker.json"))
    ap.add_argument("--reranker", default=None, help="student reranker checkpoint dir")
    ap.add_argument("--teacher-config", default=None, help="teacher_models.json to also score teacher")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--output", default=str(ROOT / "outputs" / "real-training" / "reranker-lift-report.json"))
    ap.add_argument("--run-id", default=None, help="experiment run id (run card written on success)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_reranker_config(args.config)
    if not pathlib.Path(args.candidates).exists():
        print(f"ERROR: candidates not found: {args.candidates}", file=sys.stderr)
        return 2
    rows = list(dp.stream_jsonl(args.candidates))
    fs, orc, pintk = _first_stage_and_oracle(rows, args.k)
    report = {
        "n_queries": len(rows),
        f"first_stage_ndcg@{args.k}": RM.aggregate_rows(fs).get(f"ndcg@{args.k}"),
        f"oracle_ndcg@{args.k}": RM.aggregate_rows(orc).get(f"ndcg@{args.k}"),
        f"first_stage_pos_in_top_{args.k}": round(
            sum(p[f"pos_in_top_{args.k}"] for p in pintk) / max(len(pintk), 1), 4),
    }
    print(f"[first-stage/oracle] {json.dumps(report, ensure_ascii=False)}")

    if args.dry_run:
        assert "torch" not in sys.modules, "dry-run must not import torch"
        print("dry-run-ok (no ML imports)")
        return 0

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"Reranking needs extras: pip install -e '.[train,eval]'. ({exc})")

    device = f"cuda:{args.device_index}"
    if args.reranker:
        rr_rows = []
        for r in rows:
            pairs = [(r["query"], c["document"]) for c in r["candidates"]]
            scores = RM.score_with_student_reranker(args.reranker, pairs, cfg.input_template,
                                                    device=device)
            rr_rows.append(RM.rerank_metrics([c["doc_id"] for c in r["candidates"]], scores,
                                             _positives(r), (args.k,)))
        report[f"student_reranker_ndcg@{args.k}"] = RM.aggregate_rows(rr_rows).get(f"ndcg@{args.k}")

    if args.teacher_config:
        from boldt_embed import teacher as T
        from boldt_embed.config_teacher import load_teacher_models_config
        tcfg = load_teacher_models_config(args.teacher_config)
        model = T.load_reranker_teacher(tcfg.reranker_teacher, device=device)
        tr_rows = []
        for r in rows:
            pairs = [(r["query"], c["document"]) for c in r["candidates"]]
            scores = T.score_pairs_with_reranker_teacher(model, pairs, tcfg.reranker_teacher)
            tr_rows.append(RM.rerank_metrics([c["doc_id"] for c in r["candidates"]], scores,
                                            _positives(r), (args.k,)))
        report[f"teacher_reranker_ndcg@{args.k}"] = RM.aggregate_rows(tr_rows).get(f"ndcg@{args.k}")

    report["run_metadata"] = {"command": "scripts/eval_reranker_lift.py", "commit": _git_commit(),
                              "hardware": platform.platform(), "reranker": args.reranker}
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    card = ER.emit_run_card(args.run_id, "eval", "scripts/eval_reranker_lift.py",
                            model=args.reranker, dataset=args.candidates,
                            metrics={k: v for k, v in report.items() if "ndcg@" in k},
                            input_artifacts=[args.candidates], output_artifacts=[str(out)],
                            notes="reranker lift over fixed candidate sets")
    print("=== LIFT SUMMARY ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {out}; run card: {card}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
