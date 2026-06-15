#!/usr/bin/env python3
"""Hardness-aware v5 RAG reranker lift + promotion gate (stdlib core, lazy ML only for --reranker).

Primary promotion is driven by **medium+hard** buckets on primary sets (WebFAQ/local/private RAG);
GermanQuAD/DT-test are **do-not-regress guardrails** (near-ceiling sets get a -0.005 tolerance and
are never a primary signal). The gate also blocks per-query catastrophic drops. Candidate lists are
FIXED; reranked order comes from per-candidate ``reranker_score`` (fallback ``teacher_score``), or
from a real ``--reranker`` checkpoint (lazy ML).

Usage: --primary webfaq=PATH local_rag=PATH  --guardrail germanquad=PATH dt_test=PATH
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import hardness_aware_eval as H  # noqa: E402


def _read(path: pathlib.Path) -> list:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").split("\n") if ln.strip()]


def _parse_specs(specs):
    out = []
    for s in specs or []:
        if "=" not in s:
            raise SystemExit(f"bad spec '{s}' (expected name=path)")
        name, path = s.split("=", 1)
        out.append((name, pathlib.Path(path)))
    return out


def _score_rows(rows, role, reranker, config):
    """Per-query metrics for one eval set. Uses precomputed scores unless --reranker is given."""
    if not reranker:
        return [H.list_metrics(r) for r in rows]
    from boldt_embed import reranker_modern as RM            # lazy ML
    from boldt_embed.config import load_reranker_config
    cfg = load_reranker_config(config)
    out = []
    for r in rows:
        cands = r.get("candidates") or []
        pairs = [(r.get("query", ""), c.get("text", "")) for c in cands]
        s = RM.score_with_student_reranker(reranker, pairs, cfg.input_template)
        scores = {c["doc_id"]: sc for c, sc in zip(cands, s)}
        out.append(H.list_metrics(r, reranker_scores=scores))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--primary", nargs="*", default=[], help="name=path (WebFAQ/local/private RAG)")
    ap.add_argument("--guardrail", nargs="*", default=[], help="name=path (GermanQuAD/DT-test)")
    ap.add_argument("--reranker", default=None, help="reranker checkpoint dir (real run; lazy ML)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "training_reranker.json"))
    ap.add_argument("--report", required=True)
    ap.add_argument("--primary-min-lift", type=float, default=H.PRIMARY_MIN_LIFT)
    ap.add_argument("--max-catastrophic-rate", type=float, default=H.MAX_CATASTROPHIC_RATE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    set_reports = []
    for role, specs in (("primary", args.primary), ("guardrail", args.guardrail)):
        for name, path in _parse_specs(specs):
            if not path.exists():
                print(f"ERROR: candidate lists not found: {path}", file=sys.stderr)
                return 2
            pq = _score_rows(_read(path), role, args.reranker, args.config)
            set_reports.append(H.summarize_eval_set(name, pq, role=role))

    if not set_reports:
        print("ERROR: no eval sets provided (need --primary and/or --guardrail)", file=sys.stderr)
        return 2

    gate = H.evaluate_hardness_gate(
        set_reports, primary_min_lift=args.primary_min_lift,
        max_catastrophic_rate=args.max_catastrophic_rate)
    report = {"gate": gate, "eval_sets": set_reports}

    if args.dry_run and not args.reranker:
        assert "torch" not in sys.modules, "dry-run must not import torch"

    pathlib.Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    print(f"[v5-lift] gate={gate['status']}")
    for r in set_reports:
        print(f"  {r['role']:9s} {r['eval_set']:14s} overall Δ {r['overall_delta_ndcg@10']:+.4f} "
              f"medium+hard(micro/macro) {r['primary_micro_lift']:+.4f}/{r['primary_macro_lift']:+.4f} "
              f"no_room={r['no_room_fraction']:.2f} catastrophic={r['catastrophic_rate']:.3f}")
    for c in gate["failing"]:
        print(f"  ✗ {c['check']}: {c['detail']}", file=sys.stderr)

    if gate["status"] != "pass":
        print("FAIL — v5 hardness-aware promotion gate not satisfied", file=sys.stderr)
        return 1
    print("PASS — medium+hard lift positive; guardrails within tolerance; few catastrophic drops")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
