#!/usr/bin/env python3
"""AutoResearch reporting + Pareto frontier — the read-only "where are we?" view.

Reads ONLY saved artifacts (ADR-005: a metric is only what was saved) and writes a
frontier report under ``outputs/autoresearch/reports/`` in three forms:

  - ``frontier.json``      — machine-readable everything,
  - ``frontier.md``        — human leaderboard + frontier + regressions + missing,
  - ``leaderboard.tsv``    — one row per candidate, all metrics (``missing`` where unknown).

Inputs (each best-effort; a missing input is reported as ``missing``, NEVER treated as 0):
  - ``outputs/autoresearch/results.tsv``            — per-trial dense-eval metrics,
  - ``outputs/mteb/<label>/summary.json``           — MTEB(deu) retrieval-core scores,
  - ``outputs/autoresearch/runs/<run>/metrics.json``— per-run dense metrics (richer than tsv),
  - ``outputs/autoresearch/events.jsonl``           — loop event stream (optional; may not exist),
  - merge / distill / specialist manifests          — best-effort, marked missing if absent.

Pure stdlib. NEVER trains or evals — it only aggregates what is already on disk. The Pareto
math lives in :mod:`boldt_embed.pareto`; this file is the I/O + presentation layer.

The input root is injectable (``--root`` flag, or ``build_report(root=...)``) so tests can
point it at a fixture tree instead of the repository's real ``outputs/``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the stdlib core importable whether run as a script or imported by tests.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from boldt_embed import pareto  # noqa: E402

# ---------------------------------------------------------------------------
# MTEB(deu) retrieval-core task keys + same-size peers (mirrors ar_frontier_status.py).
# ---------------------------------------------------------------------------
MTEB_TASKS = ["GermanQuAD-Retrieval", "GerDaLIRSmall", "MIRACLRetrievalHardNegatives",
              "MultiLongDocRetrieval"]
MTEB_SHORT = {"GermanQuAD-Retrieval": "MTEB_GermanQuAD", "GerDaLIRSmall": "MTEB_GerDaLIR",
              "MIRACLRetrievalHardNegatives": "MTEB_MIRACL", "MultiLongDocRetrieval": "MTEB_MLDR"}
SAME_SIZE_PEERS = ["e5-base", "lfm2.5"]
# Larger reference models that are NOT candidates and NOT same-size peers.
STRETCH_LABELS = ["qwen3-0.6b", "gte-multilingual-base"]

# The hard-target dense metrics from results.tsv / runs metrics.json (higher is better,
# except cost metrics handled by pareto).
DENSE_METRICS = ["webfaq_recall@100", "webfaq_ndcg@10", "germanquad_ndcg@10",
                 "dt_test_ndcg@10", "matryoshka_256_retention"]
COST_METRICS = ["vram_gb", "throughput_pairs_per_sec"]

# Full ordered metric list used for the leaderboard + frontier objective.
ALL_METRICS = (["webfaq_recall@100", "webfaq_ndcg@10", "germanquad_ndcg@10", "dt_test_ndcg@10"]
               + [MTEB_SHORT[t] for t in MTEB_TASKS]
               + ["matryoshka_256_retention"] + COST_METRICS)

# Hard target metrics that drive Pareto dominance (cost metrics are tie-breakers only).
HARD_TARGETS = (["webfaq_recall@100", "webfaq_ndcg@10", "germanquad_ndcg@10", "dt_test_ndcg@10"]
                + [MTEB_SHORT[t] for t in MTEB_TASKS] + ["matryoshka_256_retention"])

MISSING = "missing"


# ---------------------------------------------------------------------------
# Small artifact readers (each returns data + an explicit "missing" marker, never 0).
# ---------------------------------------------------------------------------
def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _num(v):
    """Coerce to float if it looks numeric, else None. Empty strings -> None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def read_results_tsv(path: Path):
    """Parse results.tsv into ``{run_id: {metric: float-or-None, ...meta}}``.

    Missing file -> ``({}, missing-marker)``. Each row keeps its raw ``mode``/``status``
    and the dense metrics renamed to the canonical metric keys.
    """
    if not path.exists():
        return {}, {"results_tsv": str(path), "status": MISSING}
    rows = {}
    text = path.read_text(encoding="utf-8").splitlines()
    if not text:
        return {}, {"results_tsv": str(path), "status": "empty"}
    header = text[0].split("\t")
    idx = {name: i for i, name in enumerate(header)}

    def cell(parts, name):
        i = idx.get(name)
        return parts[i] if i is not None and i < len(parts) else None

    for line in text[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        run_id = cell(parts, "run_id")
        if not run_id:
            continue
        rows[run_id] = {
            "run_id": run_id,
            "mode": cell(parts, "mode"),
            "status": cell(parts, "status"),
            "leakage_status": cell(parts, "leakage_status"),
            "config_path": cell(parts, "config_path"),
            "metrics": {
                "webfaq_recall@100": _num(cell(parts, "webfaq_recall100")),
                "webfaq_ndcg@10": _num(cell(parts, "webfaq_ndcg10")),
                "germanquad_ndcg@10": _num(cell(parts, "germanquad_ndcg10")),
                "dt_test_ndcg@10": _num(cell(parts, "dt_test_ndcg10")),
                "matryoshka_256_retention": _num(cell(parts, "m256_retention")),
                "vram_gb": _num(cell(parts, "vram_gb")),
                "throughput_pairs_per_sec": _num(cell(parts, "throughput_pairs_per_sec")),
            },
        }
    return rows, {"results_tsv": str(path), "status": "ok", "rows": len(rows)}


def read_run_metrics(run_dir: Path):
    """Read a single ``runs/<id>/metrics.json`` into canonical dense metrics, or None."""
    data = _read_json(run_dir / "metrics.json")
    if not isinstance(data, dict):
        return None
    m = data.get("metrics", {}) if isinstance(data.get("metrics"), dict) else {}
    wf = m.get("webfaq", {}) if isinstance(m.get("webfaq"), dict) else {}
    gq = m.get("germanquad", {}) if isinstance(m.get("germanquad"), dict) else {}
    dt = m.get("dt_test", {}) if isinstance(m.get("dt_test"), dict) else {}
    mat = m.get("matryoshka", {}) if isinstance(m.get("matryoshka"), dict) else {}
    sysm = m.get("system", {}) if isinstance(m.get("system"), dict) else {}
    return {
        "run_id": data.get("run_id") or run_dir.name,
        "mode": data.get("mode"),
        "status": data.get("status"),
        "leakage_status": data.get("leakage_status"),
        "metrics": {
            "webfaq_recall@100": _num(wf.get("recall@100")),
            "webfaq_ndcg@10": _num(wf.get("ndcg@10")),
            "germanquad_ndcg@10": _num(gq.get("ndcg@10")),
            "dt_test_ndcg@10": _num(dt.get("ndcg@10")),
            "matryoshka_256_retention": _num(mat.get("retention_256")),
            "vram_gb": _num(sysm.get("vram_gb")),
            "throughput_pairs_per_sec": _num(sysm.get("throughput_pairs_per_sec")),
        },
    }


def read_mteb(mteb_dir: Path, label: str):
    """Read ``outputs/mteb/<label>/summary.json`` -> ``{MTEB_SHORT: float-or-None}``.

    Missing file => every task is None (i.e. *missing*), NEVER 0.
    """
    summary = _read_json(mteb_dir / label / "summary.json")
    out = {MTEB_SHORT[t]: None for t in MTEB_TASKS}
    present = False
    if isinstance(summary, dict):
        scores = summary.get("scores", {}) or {}
        for t in MTEB_TASKS:
            v = _num(scores.get(t))
            out[MTEB_SHORT[t]] = v
            if v is not None:
                present = True
    return out, present


def read_events(path: Path):
    """Read events.jsonl best-effort. Absent file is fine -> (count, status='missing')."""
    if not path.exists():
        return 0, {"events_jsonl": str(path), "status": MISSING}
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            n += 1
    return n, {"events_jsonl": str(path), "status": "ok", "events": n}


def discover_manifests(root: Path):
    """Best-effort hunt for merge / distill / specialist manifests; mark missing if none."""
    out = {}
    for kind, patterns in (
        ("merge", ["merge_manifest.json", "slerp_manifest.json", "merge.json"]),
        ("distill", ["distill_manifest.json", "distillation_manifest.json"]),
        ("specialist", ["specialist_manifest.json", "specialists.json"]),
    ):
        found = []
        for base in (root / "outputs" / "merged", root / "outputs" / "autoresearch",
                     root / "outputs"):
            if not base.exists():
                continue
            for pat in patterns:
                found.extend(str(p) for p in base.rglob(pat))
        out[kind] = sorted(set(found)) or MISSING
    return out


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------
def _mean(vals):
    nums = [v for v in vals if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else None


def _mteb_aggregate(mteb_row):
    return _mean([mteb_row.get(MTEB_SHORT[t]) for t in MTEB_TASKS])


def build_report(root, peers=SAME_SIZE_PEERS, stretch=STRETCH_LABELS):
    """Assemble the full report dict from artifacts under ``root`` (a Path or str).

    ``root`` is the repo-style directory that contains ``outputs/`` — injectable so tests
    can point at a fixture tree. Returns a plain dict (JSON-serialisable).
    """
    root = Path(root)
    out_dir = root / "outputs"
    ar_dir = out_dir / "autoresearch"
    mteb_dir = out_dir / "mteb"

    tsv_rows, tsv_status = read_results_tsv(ar_dir / "results.tsv")
    _, events_status = read_events(ar_dir / "events.jsonl")
    manifests = discover_manifests(root)

    # ---- richer per-run dense metrics override tsv where a runs/<id>/metrics.json exists.
    runs_dir = ar_dir / "runs"
    run_metrics = {}
    if runs_dir.exists():
        for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            rm = read_run_metrics(d)
            if rm:
                run_metrics[rm["run_id"]] = rm

    # ---- dense candidates = union of tsv run_ids and runs/ ids (real runs preferred).
    dense_ids = set(tsv_rows) | set(run_metrics)
    dense_candidates = {}
    for rid in sorted(dense_ids):
        base = dict(tsv_rows.get(rid, {"run_id": rid, "metrics": {}}))
        rich = run_metrics.get(rid)
        merged = dict(base.get("metrics", {}))
        if rich:
            for k, v in rich["metrics"].items():
                if v is not None:
                    merged[k] = v
            base.setdefault("mode", rich.get("mode"))
            base.setdefault("status", rich.get("status"))
            base["leakage_status"] = base.get("leakage_status") or rich.get("leakage_status")
        dense_candidates[rid] = {
            "label": rid,
            "kind": "dense_run",
            "mode": base.get("mode"),
            "status": base.get("status"),
            "leakage_status": base.get("leakage_status"),
            "metrics": {k: merged.get(k) for k in DENSE_METRICS + COST_METRICS},
            "has_mteb": False,
        }

    # ---- MTEB candidates from outputs/mteb/<label>/summary.json (exclude peers + stretch).
    peer_set, stretch_set = set(peers), set(stretch)
    mteb_labels = []
    if mteb_dir.exists():
        mteb_labels = sorted(d.name for d in mteb_dir.iterdir()
                             if d.is_dir() and (d / "summary.json").exists())

    peer_rows = {}
    for p in peers:
        row, present = read_mteb(mteb_dir, p)
        peer_rows[p] = {"row": row, "present": present}
    peer_frontier = {}
    for t in MTEB_TASKS:
        vals = [peer_rows[p]["row"].get(MTEB_SHORT[t]) for p in peers]
        nums = [v for v in vals if isinstance(v, (int, float))]
        peer_frontier[MTEB_SHORT[t]] = max(nums) if nums else None
    peer_aggregate = _mean(list(peer_frontier.values()))

    mteb_candidates = {}
    for label in mteb_labels:
        if label in peer_set or label in stretch_set:
            continue
        row, present = read_mteb(mteb_dir, label)
        mteb_candidates[label] = {
            "label": label,
            "kind": "mteb_model",
            "metrics": {k: None for k in DENSE_METRICS + COST_METRICS},
            "mteb": row,
            "mteb_aggregate": _mteb_aggregate(row),
            "has_mteb": present,
        }

    # ---- merged candidate view: a candidate may have BOTH a dense run and an MTEB summary
    # when labels coincide; otherwise they are distinct rows. We key by label and fold MTEB
    # task scores into the metric dict so Pareto/leaderboard see one coherent vector.
    candidates = {}
    for rid, c in dense_candidates.items():
        c = dict(c)
        c["metrics"] = dict(c["metrics"])
        for t in MTEB_TASKS:
            c["metrics"].setdefault(MTEB_SHORT[t], None)
        candidates[rid] = c
    for label, c in mteb_candidates.items():
        if label in candidates:  # same label has a dense run too: fold MTEB scores in
            tgt = candidates[label]
            for t in MTEB_TASKS:
                tgt["metrics"][MTEB_SHORT[t]] = c["mteb"].get(MTEB_SHORT[t])
            tgt["mteb_aggregate"] = c["mteb_aggregate"]
            tgt["has_mteb"] = c["has_mteb"]
        else:
            row = dict(c["metrics"])
            for t in MTEB_TASKS:
                row[MTEB_SHORT[t]] = c["mteb"].get(MTEB_SHORT[t])
            candidates[label] = {
                "label": label, "kind": c["kind"], "mode": None, "status": None,
                "leakage_status": None, "metrics": row,
                "mteb_aggregate": c["mteb_aggregate"], "has_mteb": c["has_mteb"],
            }

    cand_list = list(candidates.values())

    # ---- best-by-task (over all metrics, candidates only).
    best_by_task = {}
    for m in ALL_METRICS:
        direction = pareto._direction(m, None)
        best_label, best_val = None, None
        for c in cand_list:
            v = c["metrics"].get(m)
            if not isinstance(v, (int, float)):
                continue
            if best_val is None or (v < best_val if direction == pareto.LOWER else v > best_val):
                best_val, best_label = v, c["label"]
        best_by_task[m] = {"label": best_label, "value": best_val} if best_label else MISSING

    # ---- promotable candidates: MTEB aggregate beats the same-size-peer aggregate.
    promotable = []
    for c in cand_list:
        agg = c.get("mteb_aggregate")
        if isinstance(agg, (int, float)) and isinstance(peer_aggregate, (int, float)) \
                and agg >= peer_aggregate:
            promotable.append({"label": c["label"], "mteb_aggregate": round(agg, 4),
                               "gap_to_peer": round(agg - peer_aggregate, 4)})
    promotable.sort(key=lambda x: x["mteb_aggregate"], reverse=True)

    # ---- Pareto frontier over hard targets (cost metrics are tie-breakers only).
    pareto_input = [{"label": c["label"], **{m: c["metrics"].get(m) for m in HARD_TARGETS},
                     **{m: c["metrics"].get(m) for m in COST_METRICS}} for c in cand_list]
    frontier_rows = pareto.pareto_frontier(pareto_input, metrics=HARD_TARGETS,
                                            cost_metrics=COST_METRICS)
    frontier_rows = pareto.tie_break(frontier_rows, cost_metrics=COST_METRICS)
    frontier_labels = [r["label"] for r in frontier_rows]

    # ---- regressions: dense runs whose status is fail/crash, or whose key dense metric
    # fell below the best observed real run (do-not-regress signal).
    regressions = []
    for c in cand_list:
        st = (c.get("status") or "").lower()
        if st in ("fail", "crash", "error", "rejected"):
            regressions.append({"label": c["label"], "reason": f"status={st}"})
    # also flag runs that recorded a dense webfaq recall below the best real run by >tol
    real = [c for c in cand_list
            if (c.get("mode") == "real") and isinstance(c["metrics"].get("webfaq_recall@100"),
                                                         (int, float))]
    if real:
        best_recall = max(c["metrics"]["webfaq_recall@100"] for c in real)
        for c in real:
            v = c["metrics"]["webfaq_recall@100"]
            if v < best_recall - 0.02:
                regressions.append({"label": c["label"], "reason":
                                    f"webfaq_recall@100 {round(v, 4)} < best {round(best_recall, 4)} - 0.02"})

    # ---- missing artifacts roll-up (never silently zeroed).
    missing = {
        "events_jsonl": events_status if events_status.get("status") == MISSING else "present",
        "results_tsv": tsv_status if tsv_status.get("status") == MISSING else "present",
        "manifests": {k: v for k, v in manifests.items() if v == MISSING} or "none_missing",
        "mteb_summaries_missing": sorted(c["label"] for c in cand_list if not c.get("has_mteb")),
        "peers_missing_mteb": [p for p in peers if not peer_rows[p]["present"]],
    }

    return {
        "root": str(root),
        "peers": list(peers),
        "peer_frontier": peer_frontier,
        "peer_aggregate": None if peer_aggregate is None else round(peer_aggregate, 4),
        "metric_columns": ALL_METRICS,
        "hard_target_metrics": HARD_TARGETS,
        "cost_metrics": COST_METRICS,
        "n_candidates": len(cand_list),
        "candidates": [
            {"label": c["label"], "kind": c["kind"], "mode": c.get("mode"),
             "status": c.get("status"), "leakage_status": c.get("leakage_status"),
             "has_mteb": c.get("has_mteb", False),
             "mteb_aggregate": (None if not isinstance(c.get("mteb_aggregate"), (int, float))
                                else round(c["mteb_aggregate"], 4)),
             "metrics": {m: c["metrics"].get(m) for m in ALL_METRICS}}
            for c in sorted(cand_list, key=lambda x: x["label"])
        ],
        "best_by_task": best_by_task,
        "beats_peer_aggregate": promotable,
        "pareto_frontier": frontier_labels,
        "regressions": regressions,
        "missing": missing,
        "inputs": {"results_tsv": tsv_status, "events_jsonl": events_status,
                   "manifests": manifests,
                   "mteb_labels_scanned": mteb_labels},
        "note": "Read-only aggregation of saved artifacts (ADR-005). A missing artifact is "
                "reported as 'missing', never treated as 0. Pareto dominance uses the hard "
                "target metrics; vram/throughput are tie-breakers. beats_peer_aggregate = MTEB "
                "aggregate (mean over the 4 retrieval-core tasks) >= same-size-peer (e5-base/lfm2.5) "
                "aggregate — an INDICATOR ONLY. Authoritative promotion is check_mteb_frontier_gate "
                "/ ar_promote, which ALSO enforces per-task do-not-regress, baseline presence and "
                "clean leakage; a candidate here can still fail that gate.",
    }


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------
def _fmt(v):
    if v is None or v == MISSING:
        return MISSING
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    return str(v)


def render_markdown(report: dict) -> str:
    L = []
    L.append("# AutoResearch frontier report\n")
    L.append(f"Same-size peers: {', '.join(report['peers'])}  ·  "
             f"peer MTEB aggregate: **{_fmt(report['peer_aggregate'])}**  ·  "
             f"candidates: {report['n_candidates']}\n")

    cols = report["metric_columns"]
    L.append("## Leaderboard\n")
    L.append("| candidate | kind | mode | mteb_agg | " + " | ".join(cols) + " |")
    L.append("|---|---|---|---|" + "---|" * len(cols))
    for c in report["candidates"]:
        star = " 🏆" if c["label"] in [p["label"] for p in report["beats_peer_aggregate"]] else ""
        front = " ◆" if c["label"] in report["pareto_frontier"] else ""
        row = [f"`{c['label']}`{star}{front}", c.get("kind") or "", c.get("mode") or "",
               _fmt(c.get("mteb_aggregate"))]
        row += [_fmt(c["metrics"].get(m)) for m in cols]
        L.append("| " + " | ".join(row) + " |")

    L.append("\n## Best by task\n")
    for m in cols:
        b = report["best_by_task"].get(m)
        if b == MISSING or not b:
            L.append(f"- {m}: {MISSING}")
        else:
            L.append(f"- {m}: `{b['label']}` = {_fmt(b['value'])}")

    L.append("\n## Beats same-size-peer aggregate — INDICATOR ONLY (authoritative gate = ar_promote)\n")
    if not report["beats_peer_aggregate"]:
        L.append("- none (no candidate beats the e5-base/lfm2.5 frontier aggregate)")
    for p in report["beats_peer_aggregate"]:
        L.append(f"- `{p['label']}` — MTEB agg {_fmt(p['mteb_aggregate'])} "
                 f"(gap to peer {p['gap_to_peer']:+.4f})")

    L.append("\n## Pareto frontier (hard targets; vram/throughput tie-breakers)\n")
    if not report["pareto_frontier"]:
        L.append("- empty")
    for lbl in report["pareto_frontier"]:
        L.append(f"- ◆ `{lbl}`")

    L.append("\n## Regressions\n")
    if not report["regressions"]:
        L.append("- none flagged")
    for r in report["regressions"]:
        L.append(f"- `{r['label']}`: {r['reason']}")

    L.append("\n## Missing artifacts (reported, never zeroed)\n")
    miss = report["missing"]
    L.append(f"- events.jsonl: {miss['events_jsonl'] if isinstance(miss['events_jsonl'], str) else MISSING}")
    L.append(f"- results.tsv: {miss['results_tsv'] if isinstance(miss['results_tsv'], str) else MISSING}")
    L.append(f"- manifests missing: {miss['manifests']}")
    if miss["mteb_summaries_missing"]:
        L.append(f"- candidates without an MTEB summary: {', '.join(miss['mteb_summaries_missing'])}")
    if miss["peers_missing_mteb"]:
        L.append(f"- peers without an MTEB summary: {', '.join(miss['peers_missing_mteb'])}")

    L.append("\n> " + report["note"])
    return "\n".join(L) + "\n"


def render_leaderboard_tsv(report: dict) -> str:
    cols = report["metric_columns"]
    header = ["label", "kind", "mode", "status", "has_mteb", "mteb_aggregate",
              "beats_peers", "pareto_frontier"] + cols
    promo = {p["label"] for p in report["beats_peer_aggregate"]}
    front = set(report["pareto_frontier"])
    lines = ["\t".join(header)]
    for c in report["candidates"]:
        row = [c["label"], c.get("kind") or "", c.get("mode") or "", c.get("status") or "",
               str(c.get("has_mteb", False)),
               _fmt(c.get("mteb_aggregate")),
               str(c["label"] in promo), str(c["label"] in front)]
        row += [_fmt(c["metrics"].get(m)) for m in cols]
        lines.append("\t".join(row))
    return "\n".join(lines) + "\n"


def write_reports(report: dict, root) -> dict:
    """Write frontier.json / frontier.md / leaderboard.tsv under outputs/autoresearch/reports/."""
    rep_dir = Path(root) / "outputs" / "autoresearch" / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": rep_dir / "frontier.json",
        "md": rep_dir / "frontier.md",
        "tsv": rep_dir / "leaderboard.tsv",
    }
    paths["json"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["md"].write_text(render_markdown(report), encoding="utf-8")
    paths["tsv"].write_text(render_leaderboard_tsv(report), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(_REPO_ROOT),
                    help="repo-style root containing outputs/ (injectable for tests)")
    ap.add_argument("--format", choices=["markdown", "json"], default="markdown")
    ap.add_argument("--candidate", default=None,
                    help="restrict the printed view to a single candidate label")
    ap.add_argument("--no-write", action="store_true",
                    help="do not write report files, only print to stdout")
    ap.add_argument("--peers", default=",".join(SAME_SIZE_PEERS),
                    help="same-size-peer MTEB labels (comma-separated)")
    args = ap.parse_args(argv)

    peers = [p.strip() for p in args.peers.split(",") if p.strip()]
    report = build_report(args.root, peers=peers)

    written = {}
    if not args.no_write:
        written = write_reports(report, args.root)
        report = dict(report)
        report["written"] = written

    if args.candidate:
        match = next((c for c in report["candidates"] if c["label"] == args.candidate), None)
        view = {"candidate": args.candidate,
                "found": match is not None,
                "record": match if match else MISSING,
                "on_pareto_frontier": args.candidate in report["pareto_frontier"],
                "beats_peer_aggregate": args.candidate in [p["label"] for p in report["beats_peer_aggregate"]],
                "peer_aggregate": report["peer_aggregate"],
                "written": written}
        if args.format == "json":
            print(json.dumps(view, ensure_ascii=False, indent=2))
        else:
            print(f"# candidate: {args.candidate}\n")
            if match is None:
                print(f"**{MISSING}** — no candidate with this label was found under "
                      f"{report['root']}/outputs/")
            else:
                print(f"- kind: {match.get('kind')}  ·  mode: {match.get('mode')}  ·  "
                      f"has_mteb: {match.get('has_mteb')}")
                print(f"- mteb_aggregate: {_fmt(match.get('mteb_aggregate'))} "
                      f"(peer aggregate {_fmt(report['peer_aggregate'])})")
                print(f"- on Pareto frontier: {view['on_pareto_frontier']}  ·  "
                      f"beats peer aggregate (indicator only): {view['beats_peer_aggregate']}")
                for m in report["metric_columns"]:
                    print(f"  - {m}: {_fmt(match['metrics'].get(m))}")
        return 0

    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(report))
        if written:
            print(f"\n_wrote: {written['json']}, {written['md']}, {written['tsv']}_")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
