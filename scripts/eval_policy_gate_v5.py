#!/usr/bin/env python3
"""Evaluate the FROZEN bounded rerank policy on each eval set (stdlib, no ML). For every set computes
first-stage / raw-rerank / policy nDCG@10, deltas, catastrophic-drop rate, action rates
(abstain/lock/blend/margin_override), max-downshift distribution, medium+hard lift, and no_room
delta. The promotion decision is about the POLICY, not the raw model. Reads scored candidate lists
named <set>.jsonl from --eval-dir; writes outputs/.../eval_<set>.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import bounded_rerank as BR  # noqa: E402
from boldt_embed import policy_reranker as PR  # noqa: E402
from boldt_embed.hardness_aware_eval import assign_bucket  # noqa: E402
from boldt_embed.metrics import ndcg_at_k  # noqa: E402
from boldt_embed.policy_config import load_policy  # noqa: E402

K = 10
PRIMARY = {"webfaq", "near_ceiling", "local_rag"}
GUARDRAIL = {"germanquad", "dt_test"}
DIAGNOSTIC = {"gerdalir", "legal"}


def _read(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]


def _positives(row):
    pos = set(row.get("positive_doc_ids") or [])
    if row.get("positive_doc_id"):
        pos.add(row["positive_doc_id"])
    pos |= {c.get("doc_id") for c in (row.get("candidates") or []) if c.get("is_positive")}
    pos |= {c.get("doc_id") for c in (row.get("candidates") or []) if c.get("label") == 1}
    return pos


def _normalize_first_stage(row):
    """Eval-data candidates injected beyond the first-stage list (e.g. `manual` hard negatives) carry
    no first_stage_rank — the first stage never surfaced them, so they sit at the BOTTOM of the
    first-stage order. Fill those positions (does NOT weaken the serving wrapper's production
    contract, which still requires every production candidate to be first-stage-ranked). Returns
    (normalized_row, n_unranked)."""
    cands = [dict(c) for c in (row.get("candidates") or [])]
    ranked = [c for c in cands if c.get("first_stage_rank") is not None]
    max_rank = max((float(c["first_stage_rank"]) for c in ranked), default=-1.0)
    min_score = min((float(c["first_stage_score"]) for c in cands
                     if c.get("first_stage_score") is not None), default=0.0)
    extra = 0
    for c in cands:
        if c.get("first_stage_rank") is None:
            extra += 1
            c["first_stage_rank"] = max_rank + extra
            if c.get("first_stage_score") is None:
                c["first_stage_score"] = min_score - extra
    r = dict(row)
    r["candidates"] = cands
    return r, extra


def role_of(name: str) -> str:
    if name in DIAGNOSTIC:
        return "diagnostic"
    if name in GUARDRAIL:
        return "guardrail"
    return "primary"


def eval_set(rows, policy, name, *, mode="policy_gated", allow_raw=False) -> dict:
    fs_n, raw_n, pol_n = [], [], []
    cat = 0
    downs = {}
    margin_q = lock_q = abstain_q = blend_q = 0
    unranked_total = unranked_queries = 0
    buckets = {}
    for r0 in rows:
        if not (r0.get("candidates") or []):
            continue
        r, n_unranked = _normalize_first_stage(r0)
        if n_unranked:
            unranked_total += n_unranked
            unranked_queries += 1
        cands = r["candidates"]
        pos = _positives(r)
        fs_ids = [c["doc_id"] for c in BR._first_stage(cands)]
        raw_ids = [c["doc_id"] for c in BR._reranker(cands)]
        out = PR.rerank_query(r, policy, mode=mode, allow_raw=allow_raw)
        pol_ids = [c["doc_id"] for c in sorted(out["candidates"], key=lambda c: c["final_rank"])]
        f = ndcg_at_k(fs_ids, pos, K); raw = ndcg_at_k(raw_ids, pos, K); p = ndcg_at_k(pol_ids, pos, K)
        fs_n.append(f); raw_n.append(raw); pol_n.append(p)
        if p - f <= -0.2:
            cat += 1
        d = out["diagnostics"]
        downs[d["max_downshift"]] = downs.get(d["max_downshift"], 0) + 1
        if d["margin_override_used"]:
            margin_q += 1
        if pol_ids == fs_ids:
            abstain_q += 1
        elif d["top_k_locked"] > 0:
            lock_q += 1
        if any(c["policy_action"] == "blended" for c in out["candidates"]):
            blend_q += 1
        buckets.setdefault(assign_bucket(f, ndcg_at_k(
            [x for x in fs_ids if x in pos] + [x for x in fs_ids if x not in pos], pos, K)),
            []).append((f, p))
    n = len(pol_n) or 1

    def mean(xs):
        return round(sum(xs) / len(xs), 6) if xs else 0.0
    mh = buckets.get("medium", []) + buckets.get("hard", [])
    nr = buckets.get("no_room", [])
    ranking_mode = "raw_rerank" if mode == "raw_rerank" else "policy_gated"
    return {
        "eval_set": name, "role": role_of(name), "ranking_mode": ranking_mode, "n_queries": len(pol_n),
        "first_stage_ndcg@10": mean(fs_n), "raw_rerank_ndcg@10": mean(raw_n),
        "policy_ndcg@10": mean(pol_n),
        "policy_delta": round(mean(pol_n) - mean(fs_n), 6),
        "raw_delta": round(mean(raw_n) - mean(fs_n), 6),
        "catastrophic_drop_rate": round(cat / n, 6),
        "abstain_rate": round(abstain_q / n, 6), "lock_rate": round(lock_q / n, 6),
        "blend_rate": round(blend_q / n, 6), "margin_override_rate": round(margin_q / n, 6),
        "max_downshift_distribution": dict(sorted(downs.items())),
        "medium_hard_lift": (round(sum(p - f for f, p in mh) / len(mh), 6) if mh else None),
        "no_room_delta": (round(sum(p - f for f, p in nr) / len(nr), 6) if nr else None),
        "bucket_counts": {b: len(v) for b, v in sorted(buckets.items())},
        "first_stage_unranked_candidates": unranked_total,
        "queries_with_unranked_candidates": unranked_queries,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--eval-dir", required=True, help="dir of <set>.jsonl scored candidate lists")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", choices=list(PR.MODES), default="policy_gated")
    ap.add_argument("--allow-raw-rerank-dangerous", action="store_true")
    args = ap.parse_args()

    policy = load_policy(args.policy)
    out = pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    known = sorted(PRIMARY | GUARDRAIL | DIAGNOSTIC)
    found = 0
    for name in known:
        for cand in (f"{name}.jsonl", f"{name}_scored.jsonl"):
            fp = pathlib.Path(args.eval_dir) / cand
            if fp.exists():
                rep = eval_set(_read(fp), policy, name, mode=args.mode,
                               allow_raw=args.allow_raw_rerank_dangerous)
                (out / f"eval_{name}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
                print(f"[policy-eval] {name} ({rep['role']}): policyΔ {rep['policy_delta']:+} "
                      f"rawΔ {rep['raw_delta']:+} catastrophic {rep['catastrophic_drop_rate']} "
                      f"mode {rep['ranking_mode']}")
                found += 1
                break
    assert "torch" not in sys.modules, "eval must not import torch"
    if not found:
        print(f"ERROR: no eval sets found in {args.eval_dir} (expected <set>.jsonl)", file=sys.stderr)
        return 2
    print(f"[policy-eval] wrote {found} eval reports -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
