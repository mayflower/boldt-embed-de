"""Domain-aware teacher-threshold calibration (pure stdlib).

v2 filtered every positive with a single reranker threshold (>=2.0) and fed the SAME set to the
embedder and the reranker — so the reranker trained on noisy positives (pos teacher-median ≤ neg
median in v2). v3 calibrates:

- a **threshold sweep** (acceptance at -2/0/1/2/3/4/5) overall and by domain/source/license, so
  threshold sensitivity is visible;
- **separate filters**: the embedder keeps positives at the (looser) embedder threshold (default
  2.0); the reranker keeps a **higher-precision** subset at a stricter threshold (default 4.0);
- optional **per-domain** threshold overrides;
- gates: zero unknown-license rows, real-domain accepted floors, and a cap on the suspicious
  (teacher-rejected) positive rate.

No ML, no network.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_THRESHOLDS = (-2.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
DEFAULT_EMBEDDER_THRESHOLD = 2.0
DEFAULT_RERANKER_THRESHOLD = 4.0
DEFAULT_MAX_SUSPICIOUS_RATE = 0.5
REAL_DOMAINS = ("faq_real", "admin_real", "legal_adjacency_real_no_eval_overlap")


def _row_field(r: Dict[str, Any], key: str) -> Any:
    v = r.get(key)
    if v is None and isinstance(r.get("metadata"), dict):
        v = r["metadata"].get(key)
    return v


def is_unknown_license(value: Any) -> bool:
    if value is None:
        return True
    s = str(value).strip().lower()
    return s == "" or s == "unknown"


def positive_score(row: Dict[str, Any]) -> Optional[float]:
    """The score used for acceptance: prefer reranker, else embedding."""
    s = row.get("reranker_score")
    if s is None:
        s = row.get("embedding_score")
    return None if s is None else float(s)


def is_positive(row: Dict[str, Any]) -> bool:
    return row.get("positive") is not False


def _median(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


# ----------------------------------------------------------------------- threshold sweep
def acceptance_by_threshold(rows: Sequence[Dict[str, Any]],
                            thresholds: Sequence[float] = DEFAULT_THRESHOLDS) -> Dict[str, Any]:
    positives = [r for r in rows if is_positive(r)]
    total = len(positives)
    scores = [positive_score(r) for r in positives]
    out = {}
    for t in thresholds:
        acc = sum(1 for s in scores if s is not None and s >= t)
        out[_k(t)] = {"accepted": acc, "rate": round(acc / total, 4) if total else 0.0}
    return {"total_positives": total, "by_threshold": out}


def _k(t: float) -> str:
    return str(int(t)) if float(t).is_integer() else str(t)


def acceptance_by_threshold_grouped(rows: Sequence[Dict[str, Any]], key: str,
                                    thresholds: Sequence[float] = DEFAULT_THRESHOLDS
                                    ) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(_row_field(r, key) or "unknown"), []).append(r)
    return {g: acceptance_by_threshold(rs, thresholds) for g, rs in sorted(groups.items())}


# --------------------------------------------------------------------------- thresholds
def threshold_for(domain: str, base: float, per_domain: Optional[Dict[str, float]]) -> float:
    if per_domain and domain in per_domain:
        return float(per_domain[domain])
    return base


def filter_positives(rows: Sequence[Dict[str, Any]], base: float,
                     per_domain: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if not is_positive(r):
            continue
        s = positive_score(r)
        if s is not None and s >= threshold_for(str(_row_field(r, "domain") or "unknown"),
                                                 base, per_domain):
            out.append(r)
    return out


def low_score_positives(rows: Sequence[Dict[str, Any]], threshold: float, n: int = 10
                        ) -> List[Dict[str, Any]]:
    """Positives the teacher scored BELOW threshold — suspicious (teacher disagrees with label)."""
    bad = [(positive_score(r), r) for r in rows
           if is_positive(r) and positive_score(r) is not None and positive_score(r) < threshold]
    bad.sort(key=lambda kv: kv[0])
    return [{"query": (r.get("query") or "")[:80], "domain": _row_field(r, "domain"),
             "source": _row_field(r, "source"), "reranker_score": s} for s, r in bad[:n]]


def high_score_rejected(rows: Sequence[Dict[str, Any]], threshold: float, n: int = 10
                        ) -> List[Dict[str, Any]]:
    """Non-positive rows the teacher scored HIGH — likely false negatives / mislabels."""
    good = [(positive_score(r), r) for r in rows
            if not is_positive(r) and positive_score(r) is not None and positive_score(r) >= threshold]
    good.sort(key=lambda kv: kv[0], reverse=True)
    return [{"query": (r.get("query") or "")[:80], "domain": _row_field(r, "domain"),
             "source": _row_field(r, "source"), "reranker_score": s} for s, r in good[:n]]


# --------------------------------------------------------------------------- calibrate
def calibrate(rows: Sequence[Dict[str, Any]], *, embedder_threshold: float = DEFAULT_EMBEDDER_THRESHOLD,
              reranker_threshold: float = DEFAULT_RERANKER_THRESHOLD,
              per_domain_embedder: Optional[Dict[str, float]] = None,
              per_domain_reranker: Optional[Dict[str, float]] = None,
              thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
              min_real_domain_accepted: Optional[Dict[str, int]] = None,
              max_suspicious_rate: float = DEFAULT_MAX_SUSPICIOUS_RATE,
              real_domains: Sequence[str] = REAL_DOMAINS) -> Dict[str, Any]:
    rows = list(rows)
    positives = [r for r in rows if is_positive(r)]
    embedder_kept = filter_positives(rows, embedder_threshold, per_domain_embedder)
    reranker_kept = filter_positives(rows, reranker_threshold, per_domain_reranker)

    def _by_domain(kept):
        c: Dict[str, int] = {}
        for r in kept:
            d = str(_row_field(r, "domain") or "unknown")
            c[d] = c.get(d, 0) + 1
        return dict(sorted(c.items()))

    suspicious = [r for r in positives
                  if positive_score(r) is not None
                  and positive_score(r) < threshold_for(str(_row_field(r, "domain") or "unknown"),
                                                        embedder_threshold, per_domain_embedder)]
    suspicious_rate = round(len(suspicious) / len(positives), 4) if positives else 0.0
    license_unknown = sum(1 for r in rows if is_unknown_license(_row_field(r, "license")))
    emb_by_dom = _by_domain(embedder_kept)

    # ---- gates ----
    gate_results: List[Dict[str, Any]] = []

    def _gate(name, passed, detail, domain=None):
        gate_results.append({"gate": name, "domain": domain,
                             "status": "pass" if passed else "fail", "detail": detail})

    _gate("license_unknown_rows_zero", license_unknown == 0, f"{license_unknown} unknown-license rows")
    floors = min_real_domain_accepted or {}
    for dom in real_domains:
        floor = int(floors.get(dom, 0))
        got = emb_by_dom.get(dom, 0)
        _gate("real_domain_min_accepted", got >= floor, f"embedder-accepted={got} (min {floor})",
              domain=dom)
    _gate("suspicious_positive_rate", suspicious_rate <= max_suspicious_rate,
          f"suspicious_rate={suspicious_rate} (max {max_suspicious_rate})")

    failing = [g for g in gate_results if g["status"] == "fail"]
    return {
        "status": "fail" if failing else "pass",
        "thresholds": {"embedder": embedder_threshold, "reranker": reranker_threshold,
                       "per_domain_embedder": per_domain_embedder or {},
                       "per_domain_reranker": per_domain_reranker or {}},
        "sweep_overall": acceptance_by_threshold(rows, thresholds),
        "sweep_by_domain": acceptance_by_threshold_grouped(rows, "domain", thresholds),
        "sweep_by_source": acceptance_by_threshold_grouped(rows, "source", thresholds),
        "sweep_by_license": acceptance_by_threshold_grouped(rows, "license", thresholds),
        "embedder_accepted": len(embedder_kept),
        "reranker_accepted": len(reranker_kept),
        "embedder_accepted_by_domain": emb_by_dom,
        "reranker_accepted_by_domain": _by_domain(reranker_kept),
        "median_reranker_embedder_accepted": _median([r.get("reranker_score") for r in embedder_kept]),
        "median_reranker_reranker_accepted": _median([r.get("reranker_score") for r in reranker_kept]),
        "suspicious_positive_rate": suspicious_rate,
        "suspicious_positive_count": len(suspicious),
        "license_unknown_rows": license_unknown,
        "low_score_positives": low_score_positives(rows, embedder_threshold),
        "high_score_rejected": high_score_rejected(rows, reranker_threshold),
        "gates": gate_results,
        "failing_gates": failing,
        "_embedder_kept": embedder_kept,   # used by the CLI to write the filtered files
        "_reranker_kept": reranker_kept,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    th = report["thresholds"]
    lines = ["# Teacher-threshold calibration", "",
             f"Status: **{report['status'].upper()}**", "",
             f"- embedder threshold: {th['embedder']} → kept {report['embedder_accepted']} "
             f"(median rerank {report['median_reranker_embedder_accepted']})",
             f"- reranker threshold: {th['reranker']} → kept {report['reranker_accepted']} "
             f"(median rerank {report['median_reranker_reranker_accepted']})",
             f"- suspicious positive rate: {report['suspicious_positive_rate']} · "
             f"unknown-license rows: {report['license_unknown_rows']}", "",
             "## Acceptance by threshold (positives)", "",
             "| threshold | accepted | rate |", "|--:|--:|--:|"]
    for t, v in report["sweep_overall"]["by_threshold"].items():
        lines.append(f"| {t} | {v['accepted']} | {v['rate']} |")
    lines += ["", "## Embedder vs reranker accepted by domain", "",
              "| domain | embedder | reranker |", "|---|--:|--:|"]
    doms = sorted(set(report["embedder_accepted_by_domain"]) | set(report["reranker_accepted_by_domain"]))
    for d in doms:
        lines.append(f"| {d} | {report['embedder_accepted_by_domain'].get(d, 0)} | "
                     f"{report['reranker_accepted_by_domain'].get(d, 0)} |")
    if report["failing_gates"]:
        lines += ["", "## FAILING gates"]
        for g in report["failing_gates"]:
            dom = f" [{g['domain']}]" if g.get("domain") else ""
            lines.append(f"- ❌ {g['gate']}{dom}: {g['detail']}")
    if report["low_score_positives"]:
        lines += ["", "## Low-score positives (suspicious)"]
        for e in report["low_score_positives"][:5]:
            lines.append(f"- [{e['domain']}] score={e['reranker_score']}: {e['query']}")
    if report["high_score_rejected"]:
        lines += ["", "## High-score rejected (possible false negatives)"]
        for e in report["high_score_rejected"][:5]:
            lines.append(f"- [{e['domain']}] score={e['reranker_score']}: {e['query']}")
    return "\n".join(lines) + "\n"
