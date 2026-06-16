#!/usr/bin/env python3
"""Promotion gate for the v6 reranker as the ACTUAL PRODUCT — raw lift over fixed candidate lists.
NO bounded policy, NO serving wrapper, NO abstention. A result evaluated in any policy mode is
rejected outright, and a model card that recommends a policy workaround fails the gate. This decides
whether the reranker MODEL itself is promotable. Pure stdlib.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

GATE = {
    "webfaq_min_delta": 0.05, "local_rag_min_delta": 0.03,
    "germanquad_min_delta": -0.005, "dt_test_min_delta": -0.005,
    "germanquad_max_catastrophic": 0.03, "dt_test_max_catastrophic": 0.02,
    "primary_min_positive_present_rate": 0.8,
}
DIAGNOSTIC = {"gerdalir", "legal"}
POLICY_MODE_TOKENS = ("bounded", "policy", "abstain", "margin_override")


def _is_policy_result(r) -> bool:
    mode = str(r.get("ranking_mode", "")).lower()
    path = str(r.get("result_path", "")).lower()
    if mode != "raw":
        return True
    return any(t in mode for t in POLICY_MODE_TOKENS) or any(t in path for t in POLICY_MODE_TOKENS)


def raw_reranker_gate(reports: dict, *, leakage: bool = False,
                      card_recommends_policy: bool = False) -> dict:
    checks = []

    def chk(name, ok, detail):
        checks.append({"check": name, "status": "pass" if ok else "fail", "detail": detail})

    # 1) NO policy-gated result may be used (raw only)
    for name, r in reports.items():
        if name in DIAGNOSTIC:
            continue
        if _is_policy_result(r):
            chk(f"raw_only:{name}", False,
                f"ranking_mode={r.get('ranking_mode')!r} — policy/bounded/abstain result rejected")
    chk("no_policy_gated_card", not card_recommends_policy,
        "model card must not recommend a policy workaround")
    chk("no_public_eval_leakage", not leakage, "no public-eval leakage in candidate lists")

    wf = reports.get("webfaq")
    if not wf:
        chk("webfaq_present", False, "WebFAQ primary eval missing — cannot decide promotion")
    else:
        chk("webfaq_delta", wf["delta_ndcg@10"] >= GATE["webfaq_min_delta"] - 1e-9,
            f"{wf['delta_ndcg@10']:+.4f} (min +{GATE['webfaq_min_delta']})")
        chk("webfaq_positive_present", wf["positive_present_rate"]
            >= GATE["primary_min_positive_present_rate"] - 1e-9,
            f"{wf['positive_present_rate']:.3f} (min {GATE['primary_min_positive_present_rate']})")
    lr = reports.get("local_rag")
    if lr:
        chk("local_rag_delta", lr["delta_ndcg@10"] >= GATE["local_rag_min_delta"] - 1e-9,
            f"{lr['delta_ndcg@10']:+.4f} (min +{GATE['local_rag_min_delta']})")
        chk("local_rag_positive_present", lr["positive_present_rate"]
            >= GATE["primary_min_positive_present_rate"] - 1e-9,
            f"{lr['positive_present_rate']:.3f}")
    for s in ("germanquad", "dt_test"):
        r = reports.get(s)
        if not r:
            continue
        chk(f"{s}_delta", r["delta_ndcg@10"] >= GATE[f"{s}_min_delta"] - 1e-9,
            f"{r['delta_ndcg@10']:+.4f} (min {GATE[f'{s}_min_delta']})")
        chk(f"{s}_catastrophic", r["catastrophic_drop_rate"]
            <= GATE[f"{s}_max_catastrophic"] + 1e-9,
            f"{r['catastrophic_drop_rate']:.4f} (max {GATE[f'{s}_max_catastrophic']})")

    failing = [c for c in checks if c["status"] == "fail"]
    return {"status": "pass" if not failing else "fail", "checks": checks, "failing": failing,
            "thresholds": GATE, "ignored_diagnostic_sets": sorted(set(reports) & DIAGNOSTIC),
            "evaluated_ranking_mode": "raw", "policy_gated_result_used": False}


def _load_reports(eval_dir):
    reports = {}
    for p in sorted(pathlib.Path(eval_dir).glob("*lift*.json")) or \
            sorted(pathlib.Path(eval_dir).glob("eval_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        name = r.get("eval_set") or p.stem.replace("_lift", "").replace("eval_", "")
        r.setdefault("result_path", str(p))
        reports[name] = r
    return reports


def _card_recommends_policy() -> bool:
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import validate_release_2026 as V
        card = ROOT / "model_cards" / "Boldt-Reranker-DE-350M-v1.md"
        if not card.exists():
            return False
        return bool(V.check_no_policy_gated_recommendation(card.name, card.read_text(encoding="utf-8")))
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--markdown", required=True)
    args = ap.parse_args()

    reports = _load_reports(args.eval_dir)
    if not reports:
        print(f"ERROR: no *lift*.json in {args.eval_dir}", file=sys.stderr)
        return 2
    leakage = (pathlib.Path(args.eval_dir) / "LEAKAGE").exists() or \
        any(r.get("leakage") is True for r in reports.values())
    gate = raw_reranker_gate(reports, leakage=leakage,
                             card_recommends_policy=_card_recommends_policy())

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(gate, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    md = [f"# v6 RAW reranker promotion gate: **{gate['status']}**", "",
          "_Decides whether the reranker MODEL is promotable. RAW lift over fixed candidate lists "
          "only — no bounded policy / serving wrapper / abstention. "
          f"Diagnostic (ignored): {gate['ignored_diagnostic_sets'] or 'none'}._", "",
          "| eval set | role | mode | Δ nDCG@10 | catastrophic | present |",
          "|---|---|---|--:|--:|--:|"]
    for name, r in sorted(reports.items()):
        md.append(f"| {name} | {r.get('role')} | {r.get('ranking_mode')} | "
                  f"{r.get('delta_ndcg@10'):+} | {r.get('catastrophic_drop_rate')} | "
                  f"{r.get('positive_present_rate')} |")
    md += ["", "## Checks", ""]
    for c in gate["checks"]:
        md.append(f"- {'✅' if c['status'] == 'pass' else '❌'} {c['check']}: {c['detail']}")
    md += ["", ("**Verdict: PROMOTABLE (raw reranker).**" if gate["status"] == "pass"
                else f"**Verdict: NOT promoted** — failing: {[c['check'] for c in gate['failing']]}.")]
    pathlib.Path(args.markdown).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[v6-raw-gate] status={gate['status']} failing={[c['check'] for c in gate['failing']]} "
          f"-> {args.output}")
    return 0 if gate["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
