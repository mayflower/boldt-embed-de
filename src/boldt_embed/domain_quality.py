"""Domain-quality gates: make the v2 failure mode visible BEFORE training (pure stdlib).

v2 was nominally 7-domain but the teacher rejected the synthetic admin/FAQ/legal queries
(admin 4.8%, faq 5.7% accepted), so the effective training set was ~web+wiki. v3 must block
training when that happens silently. This module compares the raw candidate set against the
teacher cache and evaluates hard gates:

- zero unknown-license rows; zero disallowed-source rows;
- each real domain (faq_real / admin_real / legal_adjacency_real_no_eval_overlap) meets a raw
  AND accepted floor;
- synthetic share within each real domain stays under a cap;
- the EFFECTIVE (teacher-accepted) distribution is not web+wiki dominated;
- per-real-domain teacher acceptance stays above a floor (else the domain fails / is flagged);
- if legal_adjacency_real accepted is below floor, we must NOT claim legal transfer from data.

No ML, no network.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

REAL_DOMAINS = ("faq_real", "admin_real", "legal_adjacency_real_no_eval_overlap")
WEB_WIKI_DOMAINS = ("web", "wiki_non_eval")
_UNCERTAIN = ("uncertain", "unknown", "verify", "tbd", "todo", "?")

DEFAULT_GATES: Dict[str, Any] = {
    "min_real_domain_accepted": {
        "faq_real": 5000, "admin_real": 5000, "legal_adjacency_real_no_eval_overlap": 5000,
    },
    "max_synthetic_share_for_real_domains": 0.25,
    "max_effective_web_wiki_share": 0.65,
    "min_teacher_acceptance_rate": 0.35,
}


def merge_gates(overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    g = {**DEFAULT_GATES, "min_real_domain_accepted": dict(DEFAULT_GATES["min_real_domain_accepted"])}
    for k, v in (overrides or {}).items():
        if k == "min_real_domain_accepted" and isinstance(v, dict):
            g["min_real_domain_accepted"].update(v)
        else:
            g[k] = v
    return g


def is_unknown_license(value: Any) -> bool:
    if value is None:
        return True
    s = str(value).strip().lower()
    return s == "" or s == "unknown"


def _row_field(r: Dict[str, Any], key: str) -> Any:
    v = r.get(key)
    if v is None and isinstance(r.get("metadata"), dict):
        v = r["metadata"].get(key)
    return v


def is_synthetic_row(row: Dict[str, Any], supplemental_sources: Set[str]) -> bool:
    """A row is synthetic/supplemental if flagged, inherited-license, from a supplemental source,
    or its source id is named synthetic/generated."""
    if row.get("synthetic") is True or row.get("generated") is True:
        return True
    if _row_field(row, "license_origin") == "inherited":
        return True
    src = str(_row_field(row, "source") or _row_field(row, "source_id") or "")
    if src in supplemental_sources:
        return True
    return src.startswith("synthetic") or "synthetic" in src or "generated" in src


def _accepted(row: Dict[str, Any], threshold: float) -> bool:
    """Teacher-validated positive: positive (default True) with reranker score >= threshold."""
    if row.get("positive") is False:
        return False
    rs = row.get("reranker_score")
    if rs is None:
        rs = row.get("embedding_score")
    return rs is not None and float(rs) >= threshold


def _count_by(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        v = str(_row_field(r, key) or "unknown")
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def _median(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


def analyze(candidates: Sequence[Dict[str, Any]], cache_rows: Sequence[Dict[str, Any]],
            gates: Optional[Dict[str, Any]] = None, reranker_threshold: float = 2.0,
            supplemental_sources: Optional[Set[str]] = None,
            disallowed_sources: Optional[Set[str]] = None) -> Dict[str, Any]:
    g = merge_gates(gates)
    supplemental_sources = set(supplemental_sources or ())
    disallowed_sources = set(disallowed_sources or ())

    # (1) raw candidates by domain/source/license
    raw_by_domain = _count_by(candidates, "domain")
    # (2) accepted (teacher-validated) by domain/source/license
    accepted_rows = [r for r in cache_rows if _accepted(r, reranker_threshold)]
    acc_by_domain = _count_by(accepted_rows, "domain")
    total_accepted = len(accepted_rows)

    domains = sorted(set(raw_by_domain) | set(acc_by_domain)
                     | set(_count_by(cache_rows, "domain")))
    per_domain: Dict[str, Any] = {}
    for dom in domains:
        dom_cache = [r for r in cache_rows if str(_row_field(r, "domain") or "unknown") == dom]
        dom_acc = [r for r in accepted_rows if str(_row_field(r, "domain") or "unknown") == dom]
        raw = raw_by_domain.get(dom, len(dom_cache))
        accepted = len(dom_acc)
        syn_acc = sum(1 for r in dom_acc if is_synthetic_row(r, supplemental_sources))
        suspicious = sum(1 for r in dom_cache
                         if r.get("positive") is not False and not _accepted(r, reranker_threshold))
        per_domain[dom] = {
            "raw": raw,
            "accepted": accepted,
            "acceptance_rate": round(accepted / raw, 4) if raw else 0.0,
            "synthetic_accepted": syn_acc,
            "real_accepted": accepted - syn_acc,
            "synthetic_share": round(syn_acc / accepted, 4) if accepted else 0.0,
            "effective_share": round(accepted / total_accepted, 4) if total_accepted else 0.0,
            "median_embedding_score": _median([r.get("embedding_score") for r in dom_cache]),
            "median_reranker_score": _median([r.get("reranker_score") for r in dom_cache]),
            "suspicious_positives": suspicious,
        }

    # (9)/(10) provenance gates
    license_unknown_rows = sum(1 for r in cache_rows if is_unknown_license(_row_field(r, "license")))
    disallowed_rows = sum(1 for r in cache_rows
                          if _row_field(r, "allowed_for_training") is False
                          or str(_row_field(r, "source") or "") in disallowed_sources)
    web_wiki_accepted = sum(acc_by_domain.get(d, 0) for d in WEB_WIKI_DOMAINS)
    effective_web_wiki_share = round(web_wiki_accepted / total_accepted, 4) if total_accepted else 0.0

    # -------- gate evaluation --------
    gate_results: List[Dict[str, Any]] = []

    def _gate(name, passed, detail, domain=None):
        gate_results.append({"gate": name, "domain": domain,
                             "status": "pass" if passed else "fail", "detail": detail})

    _gate("license_unknown_rows_zero", license_unknown_rows == 0,
          f"{license_unknown_rows} unknown-license rows")
    _gate("disallowed_source_rows_zero", disallowed_rows == 0,
          f"{disallowed_rows} disallowed-source rows")

    floors = g["min_real_domain_accepted"]
    for dom in REAL_DOMAINS:
        info = per_domain.get(dom, {"raw": 0, "accepted": 0, "acceptance_rate": 0.0,
                                    "synthetic_share": 0.0})
        floor = int(floors.get(dom, 0))
        _gate("real_domain_min_raw", info["raw"] >= floor,
              f"raw={info['raw']} (min {floor})", domain=dom)
        _gate("real_domain_min_accepted", info["accepted"] >= floor,
              f"accepted={info['accepted']} (min {floor})", domain=dom)
        _gate("real_domain_synthetic_share", info["synthetic_share"] <= g["max_synthetic_share_for_real_domains"],
              f"synthetic_share={info['synthetic_share']} (max {g['max_synthetic_share_for_real_domains']})",
              domain=dom)
        _gate("real_domain_acceptance_rate", info["acceptance_rate"] >= g["min_teacher_acceptance_rate"],
              f"acceptance_rate={info['acceptance_rate']} (min {g['min_teacher_acceptance_rate']})",
              domain=dom)

    _gate("effective_web_wiki_share", effective_web_wiki_share <= g["max_effective_web_wiki_share"],
          f"web+wiki share={effective_web_wiki_share} (max {g['max_effective_web_wiki_share']})")

    # warnings (non-blocking): non-real domains below the acceptance floor -> flag for review
    review_flags: List[Dict[str, Any]] = []
    for dom, info in per_domain.items():
        if dom not in REAL_DOMAINS and info["accepted"] and \
                info["acceptance_rate"] < g["min_teacher_acceptance_rate"]:
            review_flags.append({"domain": dom, "acceptance_rate": info["acceptance_rate"],
                                 "reason": "below_min_teacher_acceptance_rate"})

    legal = per_domain.get("legal_adjacency_real_no_eval_overlap", {"accepted": 0})
    legal_floor = int(floors.get("legal_adjacency_real_no_eval_overlap", 0))
    can_claim_legal = legal["accepted"] >= legal_floor

    failing = [r for r in gate_results if r["status"] == "fail"]
    status = "fail" if failing else "pass"
    return {
        "status": status,
        "reranker_threshold": reranker_threshold,
        "gates_config": g,
        "totals": {"raw_candidates": len(candidates), "cache_rows": len(cache_rows),
                   "accepted_positives": total_accepted,
                   "license_unknown_rows": license_unknown_rows,
                   "disallowed_source_rows": disallowed_rows,
                   "effective_web_wiki_share": effective_web_wiki_share},
        "raw_by_domain": raw_by_domain,
        "raw_by_source": _count_by(candidates, "source"),
        "raw_by_license": _count_by(candidates, "license"),
        "accepted_by_domain": acc_by_domain,
        "accepted_by_source": _count_by(accepted_rows, "source"),
        "accepted_by_license": _count_by(accepted_rows, "license"),
        "per_domain": per_domain,
        "gates": gate_results,
        "failing_gates": failing,
        "review_flags": review_flags,
        "can_claim_legal_transfer_from_data": can_claim_legal,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    t = report["totals"]
    lines = ["# v3 domain-quality report", "",
             f"Status: **{report['status'].upper()}**", "",
             f"- raw candidates: {t['raw_candidates']} · cache rows: {t['cache_rows']} · "
             f"accepted positives: {t['accepted_positives']}",
             f"- unknown-license rows: {t['license_unknown_rows']} · disallowed-source rows: "
             f"{t['disallowed_source_rows']}",
             f"- effective web+wiki share: {t['effective_web_wiki_share']} "
             f"(max {report['gates_config']['max_effective_web_wiki_share']})",
             f"- can claim legal transfer from data: "
             f"**{report['can_claim_legal_transfer_from_data']}**", "",
             "## Per-domain", "",
             "| domain | raw | accepted | accept% | synth share | eff share | med rerank | suspicious |",
             "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for dom in sorted(report["per_domain"]):
        d = report["per_domain"][dom]
        lines.append(f"| {dom} | {d['raw']} | {d['accepted']} | {d['acceptance_rate']} | "
                     f"{d['synthetic_share']} | {d['effective_share']} | "
                     f"{d['median_reranker_score']} | {d['suspicious_positives']} |")
    failing = report["failing_gates"]
    lines += ["", "## Gates", ""]
    if not failing:
        lines.append("All gates **passed**.")
    else:
        lines.append("**FAILING gates:**")
        for r in failing:
            dom = f" [{r['domain']}]" if r.get("domain") else ""
            lines.append(f"- ❌ {r['gate']}{dom}: {r['detail']}")
    if report["review_flags"]:
        lines += ["", "## Review flags (non-blocking)"]
        for f in report["review_flags"]:
            lines.append(f"- ⚠️ {f['domain']}: {f['reason']} (acceptance {f['acceptance_rate']})")
    return "\n".join(lines) + "\n"
