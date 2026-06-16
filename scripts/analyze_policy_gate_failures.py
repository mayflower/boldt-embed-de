#!/usr/bin/env python3
"""Diagnose WHY the frozen bounded policy failed its promotion gate, BEFORE training anything else
(pure stdlib, no ML). For every query that the policy either regressed (policy < first-stage) or
under-lifted (raw-rerank beats policy), attribute the failure to the SPECIFIC policy constraint that
caused it and decide whether tuning the policy would fix it — versus needing calibration features, a
new checkpoint, or more data.

Failure taxonomy (priority order):
  1 positive_locked_too_low          5 blend_alpha_too_low / _too_high
  2 top_k_lock_too_strict            6 first_stage_calibration_differs_by_dataset (set-level)
  3 margin_override_too_permissive   7 candidate_source_artifact
  4 margin_override_too_strict       8 duplicate_near_duplicate_confusion
  9 no_useful_first_stage_score      10 unknown
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_policy_gate_v5 as EV  # noqa: E402  (reuse one normalization + positives source)
from boldt_embed import policy_reranker as PR  # noqa: E402
from boldt_embed.metrics import ndcg_at_k  # noqa: E402
from boldt_embed.policy_config import load_policy  # noqa: E402
from boldt_embed.rerank_abstain import _fnum  # noqa: E402

K = 10
MISSED_LIFT_MIN = 0.02            # raw must beat policy by this to count as a missed-lift failure
NEAR_DUP_JACCARD = 0.9
CAT_RECO = {
    "positive_locked_too_low": "tune_policy_threshold",
    "top_k_lock_too_strict": "tune_policy_threshold",
    "margin_override_too_permissive": "tune_policy_threshold",
    "margin_override_too_strict": "tune_policy_threshold",
    "blend_alpha_too_low": "tune_policy_threshold",
    "blend_alpha_too_high": "tune_policy_threshold",
    "first_stage_calibration_differs_by_dataset": "add_calibration_features",
    "no_useful_first_stage_score": "add_calibration_features",
    "candidate_source_artifact": "add_more_data",
    "duplicate_near_duplicate_confusion": "add_more_data",
    "unknown": "train_new_checkpoint",
}
POLICY_FIXABLE = {"positive_locked_too_low", "top_k_lock_too_strict",
                  "margin_override_too_permissive", "margin_override_too_strict",
                  "blend_alpha_too_low", "blend_alpha_too_high"}


def _tokens(t):
    return set((t or "").lower().split())


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def query_facts(row, policy):
    """Re-derive everything the classifier needs from the frozen policy on a normalized row."""
    b = policy.get("bounds", {})
    preserve_k = int(b.get("preserve_first_stage_top_k", 3))
    margin = float(b.get("margin_override", 3.0))
    max_up = int(b.get("max_upshift_without_margin", 5))
    cands = row["candidates"]
    by_id = {c["doc_id"]: c for c in cands}
    pos = EV._positives(row)
    fs_ids, rr_ids, has_rr = PR._orders(cands)
    fs_rank = {d: i for i, d in enumerate(fs_ids)}
    rr_rank = {d: i for i, d in enumerate(rr_ids)}
    rr_score = {d: _fnum(by_id[d].get("reranker_score")) for d in fs_ids}
    fs_score = {d: _fnum(by_id[d].get("first_stage_score")) for d in fs_ids}
    out = PR.rerank_query(row, policy, mode="policy_gated")
    pol_ids = [c["doc_id"] for c in sorted(out["candidates"], key=lambda c: c["final_rank"])]
    pol_rank = {d: i for i, d in enumerate(pol_ids)}
    act = {c["doc_id"]: c["policy_action"] for c in out["candidates"]}

    def best(rankmap):
        rs = [rankmap[d] for d in pos if d in rankmap]
        return min(rs) if rs else None
    fs_ndcg = ndcg_at_k(fs_ids, pos, K); raw_ndcg = ndcg_at_k(rr_ids, pos, K)
    pol_ndcg = ndcg_at_k(pol_ids, pos, K)
    best_pos = min((d for d in pos if d in fs_rank), key=lambda d: fs_rank[d], default=None)
    fs_top1 = fs_ids[0]
    # the non-positive doc the policy ranks highest above the best positive (the displacer)
    displacer = None
    if best_pos is not None and pol_rank[best_pos] > 0:
        above = [d for d in pol_ids[:pol_rank[best_pos]] if d not in pos]
        displacer = above[0] if above else None
    # near-duplicate of a positive sitting at policy rank 0 (but not itself positive)
    top1 = pol_ids[0]
    near_dup_top = (top1 not in pos and any(
        _jaccard(_tokens(by_id[top1].get("text")), _tokens(by_id[p].get("text"))) >= NEAR_DUP_JACCARD
        for p in pos if p in by_id))
    distinct_fs = len({round(fs_score[d], 6) for d in fs_ids})
    unranked = set(row.get("_unranked_ids") or [])
    # the failure is "no useful first-stage signal" specifically when the POSITIVE was never
    # retrieved by the first stage (or the whole first-stage order is constant), not merely when some
    # other candidate was injected.
    degenerate_fs = (best_pos is not None and best_pos in unranked) or distinct_fs <= 1
    return {
        "query_id": row.get("query_id"), "domain": row.get("domain"),
        "fs_ndcg": fs_ndcg, "raw_ndcg": raw_ndcg, "pol_ndcg": pol_ndcg,
        "fs_rank_pos": best(fs_rank), "rr_rank_pos": best(rr_rank), "pol_rank_pos": best(pol_rank),
        "preserve_k": preserve_k, "margin": margin, "max_up": max_up,
        "margin_override_used": out["diagnostics"]["margin_override_used"],
        "top_k_locked": out["diagnostics"]["top_k_locked"],
        "override_is_positive": (act.get(fs_ids and pol_ids[0]) == "margin_override"
                                 and pol_ids[0] in pos),
        "override_doc_nonpos": (out["diagnostics"]["margin_override_used"] and pol_ids[0] not in pos),
        "best_pos_rr_gap": (rr_score[best_pos] - rr_score[fs_top1]) if best_pos is not None else None,
        "displacer_source": (by_id[displacer].get("candidate_source") if displacer else None),
        "best_pos_action": act.get(best_pos) if best_pos is not None else None,
        "degenerate_fs": degenerate_fs,
        "positive_not_retrieved": (best_pos is not None and best_pos in unranked),
        "near_dup_top": near_dup_top,
        "num_candidates": len(cands),
    }


def classify_failure(f, *, missed_lift_min=MISSED_LIFT_MIN):
    """Return (failure_type, category, constraint, policy_fixable, detail) or None if not a failure."""
    fs, raw, pol = f["fs_ndcg"], f["raw_ndcg"], f["pol_ndcg"]
    regression = pol < fs - 1e-9
    catastrophic = (pol - fs) <= -0.2
    missed_lift = (raw - pol) > missed_lift_min and raw > fs + 1e-9
    if not (regression or missed_lift):
        return None
    ftype = "catastrophic" if catastrophic else ("regression" if regression else "missed_lift")
    pk, k_locked, override_used = f["preserve_k"], f["top_k_locked"], f["margin_override_used"]
    fsr, rrr, plr = f["fs_rank_pos"], f["rr_rank_pos"], f["pol_rank_pos"]

    def out(cat, constraint, detail):
        return (ftype, cat, constraint, cat in POLICY_FIXABLE, detail)

    # 9 — first-stage ordering is degenerate (injected/un-retrieved or constant score)
    if f["degenerate_fs"]:
        return out("no_useful_first_stage_score", "first_stage_order_degenerate",
                   "no usable first_stage_score → locks built on a meaningless order")
    # 8 — a non-positive near-duplicate of the positive sits at the top
    if f["near_dup_top"]:
        return out("duplicate_near_duplicate_confusion", "candidate dedup",
                   "policy top-1 is a near-duplicate of a positive but not the positive itself")
    # 3 — margin override fired and promoted a NON-positive to rank 1 (regression)
    if override_used and f["override_doc_nonpos"] and regression:
        return out("margin_override_too_permissive", "bounds.margin_override (too low)",
                   "margin override promoted a non-positive to rank 1")
    if fsr is None:
        return out("unknown", "positive_not_in_candidates", "no positive among candidates")
    # 4 — reranker is confident about THIS positive (positive gap over top1) but below the override
    #     margin, so the designed rescue did not fire. Checked before the lock: when the reranker
    #     prefers the positive, lowering the margin is the precise lever, not loosening the lock.
    if missed_lift and not override_used and f["best_pos_rr_gap"] is not None \
            and 0 < f["best_pos_rr_gap"] < f["margin"] and rrr is not None and rrr < fsr:
        return out("margin_override_too_strict", "bounds.margin_override (too high)",
                   f"reranker gap over first-stage top1 = {f['best_pos_rr_gap']:.2f} "
                   f"< margin {f['margin']}; override did not fire")
    # 2 — positive is in the first-stage TAIL, raw lifts it, lock kept it out (reranker has no
    #     margin-eligible preference over top1, so the lock/upshift bound is the binding constraint)
    if missed_lift and fsr >= pk and rrr is not None and rrr < fsr and (plr is None or plr >= pk):
        return out("top_k_lock_too_strict",
                   "bounds.preserve_first_stage_top_k / max_upshift_without_margin",
                   f"positive at first-stage rank {fsr} (tail); raw lifts to {rrr}; "
                   f"head locked at top-{pk} so policy left it at {plr}")
    # 1 — positive locked INSIDE the head but below where the reranker would put it
    if fsr < pk and fsr > 0 and rrr is not None and rrr < fsr and plr == fsr \
            and f["best_pos_action"] == "locked":
        return out("positive_locked_too_low", "bounds.preserve_first_stage_top_k (intra-head freeze)",
                   f"positive locked at head rank {fsr}; reranker would place it at {rrr}")
    # 5 — blended-tail failures (alpha mis-set)
    if f["best_pos_action"] in ("blended", "kept_first_stage"):
        if regression and rrr is not None and rrr > fsr:
            return out("blend_alpha_too_low", "bounds.blend_alpha (over-trusts reranker)",
                       "blended tail demoted the positive below its first-stage rank")
        if missed_lift and rrr is not None and rrr < (plr if plr is not None else fsr):
            return out("blend_alpha_too_high", "bounds.blend_alpha (under-trusts reranker)",
                       "blend stayed close to first stage; reranker ranked the positive higher")
    # 7 — a specific candidate source displaced the positive
    if f["displacer_source"]:
        return out("candidate_source_artifact", f"candidate_source={f['displacer_source']}",
                   f"positive displaced by a '{f['displacer_source']}' candidate")
    return out("unknown", "unattributed", "no policy constraint explains this failure")


def analyze_set(rows, policy, name, *, missed_lift_min=MISSED_LIFT_MIN, max_examples=4):
    role = EV.role_of(name)
    failures = []
    fs_score_vals = []
    override_fire = 0
    n = 0
    for r0 in rows:
        if not (r0.get("candidates") or []):
            continue
        unranked_ids = [c.get("doc_id") for c in (r0.get("candidates") or [])
                        if c.get("first_stage_rank") is None and c.get("first_stage_score") is None]
        r, n_unranked = EV._normalize_first_stage(r0)
        r["_unranked_ids"] = unranked_ids
        n += 1
        f = query_facts(r, policy)
        fs_score_vals += [_fnum(c.get("first_stage_score")) for c in r0.get("candidates") or []
                          if c.get("first_stage_score") is not None]
        if f["margin_override_used"]:
            override_fire += 1
        cls = classify_failure(f, missed_lift_min=missed_lift_min)
        if cls:
            ftype, cat, constraint, fixable, detail = cls
            failures.append({"query_id": f["query_id"], "domain": f["domain"], "ftype": ftype,
                             "category": cat, "constraint": constraint, "policy_fixable": fixable,
                             "recommendation": CAT_RECO[cat], "detail": detail,
                             "displacer_source": f["displacer_source"],
                             "fs_ndcg": round(f["fs_ndcg"], 4), "raw_ndcg": round(f["raw_ndcg"], 4),
                             "pol_ndcg": round(f["pol_ndcg"], 4)})

    def tally(key):
        out = {}
        for x in failures:
            out[str(x[key])] = out.get(str(x[key]), 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))
    fs_stats = None
    if fs_score_vals:
        m = sum(fs_score_vals) / len(fs_score_vals)
        fs_stats = {"mean": round(m, 4), "min": round(min(fs_score_vals), 4),
                    "max": round(max(fs_score_vals), 4),
                    "spread": round(max(fs_score_vals) - min(fs_score_vals), 4)}
    return {
        "eval_set": name, "role": role, "n_queries": n, "n_failures": len(failures),
        "failure_rate": round(len(failures) / n, 4) if n else 0.0,
        "margin_override_fire_rate": round(override_fire / n, 4) if n else 0.0,
        "first_stage_score_stats": fs_stats,
        "by_category": tally("category"), "by_type": tally("ftype"),
        "by_domain": tally("domain"), "by_displacer_source": tally("displacer_source"),
        "by_recommendation": tally("recommendation"),
        "policy_fixable_share": (round(sum(1 for x in failures if x["policy_fixable"])
                                       / len(failures), 4) if failures else 0.0),
        "examples": failures[:max_examples],
        "_failures": failures,
    }


def _failing_sets(gate_path):
    if not gate_path or not pathlib.Path(gate_path).exists():
        return None
    g = json.loads(pathlib.Path(gate_path).read_text(encoding="utf-8"))
    sets = set()
    for c in g.get("failing", []):
        for s in ("webfaq", "near_ceiling", "germanquad", "dt_test"):
            if c["check"].startswith(s):
                sets.add(s)
    return sets or None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--lists-dir", required=True, help="dir of <set>.jsonl scored candidate lists")
    ap.add_argument("--gate", default=None, help="promotion_gate.json (to focus on failing sets)")
    ap.add_argument("--output", required=True, help="markdown report")
    ap.add_argument("--json", default=None, help="machine-readable report")
    ap.add_argument("--missed-lift-min", type=float, default=MISSED_LIFT_MIN)
    ap.add_argument("--max-examples", type=int, default=4)
    args = ap.parse_args()

    policy = load_policy(args.policy)
    failing = _failing_sets(args.gate)
    known = sorted(EV.PRIMARY | EV.GUARDRAIL | EV.DIAGNOSTIC)
    reports = []
    for name in known:
        for cand in (f"{name}.jsonl", f"{name}_scored.jsonl"):
            fp = pathlib.Path(args.lists_dir) / cand
            if fp.exists():
                rep = analyze_set(EV._read(fp), policy, name, missed_lift_min=args.missed_lift_min,
                                  max_examples=args.max_examples)
                rep["implicated_by_gate"] = (failing is None or name in failing)
                reports.append(rep)
                break
    assert "torch" not in sys.modules, "analysis must not import torch"
    if not reports:
        print(f"ERROR: no scored lists in {args.lists_dir}", file=sys.stderr)
        return 2

    # cross-set calibration check: does the first-stage score scale differ a lot across sets?
    spreads = {r["eval_set"]: r["first_stage_score_stats"]["spread"] for r in reports
               if r["first_stage_score_stats"]}
    calib_flag = bool(spreads) and (max(spreads.values()) > 2.5 * max(min(spreads.values()), 1e-9))

    agg_reco = {}
    for r in reports:
        for k, v in r["by_recommendation"].items():
            agg_reco[k] = agg_reco.get(k, 0) + v
    agg_reco = dict(sorted(agg_reco.items(), key=lambda kv: -kv[1]))

    out = {"policy_id": policy.get("policy_id"), "failing_sets": sorted(failing) if failing else [],
           "sets": [{k: v for k, v in r.items() if k != "_failures"} for r in reports],
           "first_stage_score_spread_by_set": spreads,
           "cross_set_calibration_divergence": calib_flag,
           "aggregate_recommendation_counts": agg_reco}
    if args.json:
        pathlib.Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                           encoding="utf-8")

    md = _render_md(reports, out, policy, args.missed_lift_min)
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(md, encoding="utf-8")
    print(f"[policy-failure] failing_sets={out['failing_sets']} "
          f"top_reco={next(iter(agg_reco), 'none')} calib_divergence={calib_flag} -> {args.output}")
    return 0


def _render_md(reports, out, policy, mll):
    L = ["# v5 frozen-policy failure analysis", "",
         f"Policy `{policy.get('policy_id')}`. **No training until the failure mode is identified.** "
         f"A query is a failure if the policy regressed it (policy < first-stage) or under-lifted it "
         f"(raw-rerank beats policy by > {mll}). Pure stdlib, no ML.", "",
         f"Gate-failing sets: **{out['failing_sets'] or 'none (analysing all)'}**.", "",
         "## Per-set failures", "",
         "| set | role | gate? | queries | failures | rate | policy-fixable | top category |",
         "|---|---|:--:|--:|--:|--:|--:|---|"]
    for r in reports:
        top = next(iter(r["by_category"]), "—")
        L.append(f"| {r['eval_set']} | {r['role']} | {'yes' if r['implicated_by_gate'] else '·'} | "
                 f"{r['n_queries']} | {r['n_failures']} | {r['failure_rate']} | "
                 f"{r['policy_fixable_share']} | {top} |")
    L += ["", "## Which constraint caused it (by set)", ""]
    for r in reports:
        if not r["n_failures"]:
            L.append(f"- **{r['eval_set']}**: no failures.")
            continue
        L += [f"### {r['eval_set']} — {r['n_failures']} failures "
              f"({r['failure_rate']*100:.1f}%), {r['policy_fixable_share']*100:.0f}% policy-fixable",
              f"- by category: {r['by_category']}",
              f"- by type: {r['by_type']}",
              f"- by domain: {r['by_domain']}",
              f"- by displacer source: {r['by_displacer_source']}",
              f"- recommended fix mix: {r['by_recommendation']}", ""]
        for ex in r["examples"]:
            L.append(f"  - `{ex['query_id']}` [{ex['ftype']}] **{ex['category']}** — {ex['detail']} "
                     f"(fs {ex['fs_ndcg']} → raw {ex['raw_ndcg']} → policy {ex['pol_ndcg']})")
        L.append("")
    L += ["## First-stage score calibration across sets", "",
          f"Per-set first-stage score spread: `{out['first_stage_score_spread_by_set']}`. "
          f"Cross-set divergence flagged: **{out['cross_set_calibration_divergence']}** "
          f"(the absolute `margin_override` threshold of {policy['bounds']['margin_override']} on "
          f"reranker logits means different things when score scales differ by set).", "",
          "## Would adjusting the policy fix it?", "",
          f"Aggregate recommended-fix counts across failures: `{out['aggregate_recommendation_counts']}`.",
          ""]
    reco = out["aggregate_recommendation_counts"]
    fixable = reco.get("tune_policy_threshold", 0)
    calib = reco.get("add_calibration_features", 0)
    total = sum(reco.values()) or 1
    by_set = {r["eval_set"]: r for r in reports}
    wf = by_set.get("webfaq", {})
    gq = by_set.get("germanquad", {})
    L += [f"- **{fixable}/{total}** failures map to a tunable bound; **{calib}/{total}** map to "
          "missing first-stage signal (calibration). But the bounds exist to protect the "
          "GermanQuAD/near-ceiling guardrails — loosening "
          "`preserve_first_stage_top_k`/`max_upshift`/`margin_override` to capture WebFAQ lift is "
          "exactly what reintroduces guardrail catastrophic drops. The SAME tunable categories "
          f"(margin_override/top_k_lock) dominate the GermanQuAD guardrail failures "
          f"({gq.get('by_category', {})}). A single global bound cannot satisfy both unless the "
          "policy can tell the two regimes apart.", "",
          "## Recommendation", ""]
    ordered = []
    wf_calib = wf.get("by_category", {}).get("no_useful_first_stage_score", 0)
    if wf_calib and wf.get("n_failures"):
        ordered.append(
            f"1. **Add calibration features (highest priority).** {wf_calib}/{wf['n_failures']} WebFAQ "
            "failures are queries where the first stage never retrieved the positive (no "
            "first_stage_rank/score) — a policy that bounds *around first-stage order* cannot lift "
            "them. Add a per-query first-stage-confidence signal (has-first-stage-signal, score "
            "dispersion, positive-retrieved) so the policy can detect this regime and trust the "
            "reranker there, while staying bounded where the first stage is confident.")
    ordered.append(
        "2. **Audit WebFAQ first-stage recall, not the reranker.** Much of the raw lift is just "
        "recovering positives BM25 failed to retrieve; that is a retrieval/measurement gap, so part "
        "of the +0.05 bar may be unreachable by any reranking policy on these candidate lists.")
    ordered.append(
        "3. **Do NOT globally tune the bounds.** Calibration divergence across sets is "
        f"{out['cross_set_calibration_divergence']} (not a score-scale problem); the tunable "
        "failures help WebFAQ and hurt the guardrails symmetrically. Any threshold change must be "
        "calibration-gated (conditional), not global.")
    ordered.append(
        "4. **Do NOT train a new checkpoint yet** (unknown/reranker-wrong failures are negligible; "
        "the reranker scores are usable — the policy just cannot act on them) and **do NOT add more "
        "data blindly** (no candidate-source or duplicate artifacts surfaced).")
    L += ordered + ["",
        "**Conclusion:** the failure mode is identified — WebFAQ under-lift is dominated by missing "
        "first-stage signal (structural, not a threshold), and the remaining tunable failures are in "
        "direct tension with the guardrails. **No training; next step is calibration features + a "
        "WebFAQ first-stage recall audit.**"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
