#!/usr/bin/env python3
"""v2 comparison report: decide whether the data-scale run improved the project. Pure stdlib.

Reads v1 (outputs/baselines/real_*.json, real_bimntp_*.json, real_matryoshka_*.json) and v2
(<v2-dir>/eval/dense_*.json, reranker-lift-*-v2.json) results, evaluates them against the v2
success criteria, and writes V2_RESULTS.md + .json with an executive verdict (improved/mixed/
failed) and auto-generated recommendations for any failed criterion. Missing files are warned,
not fatal.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed.v2_experiment_config import load_v2_experiment_config  # noqa: E402

DATASETS = ["germanquad", "dt_test", "gerdalir"]


def _load(path):
    p = pathlib.Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ndcg_by_model(report):
    """{model_basename: ndcg@10} from a run_baseline_benchmarks report."""
    out = {}
    for r in (report or {}).get("results", []):
        m = r.get("metrics") or {}
        if "ndcg@10" in m:
            out[str(r.get("model", "")).split("/")[-1]] = m["ndcg@10"]
    return out


def _lift_delta(report):
    if not report:
        return None
    fs = next((v for k, v in report.items() if k.startswith("first_stage_ndcg@")), None)
    rr = next((v for k, v in report.items() if k.startswith("student_reranker_ndcg@")), None)
    return None if fs is None or rr is None else round(rr - fs, 4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v1-dir", default=str(ROOT / "outputs"))
    ap.add_argument("--v2-dir", default=str(ROOT / "outputs" / "v2-generalization"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiments" / "v2_generalization.json"))
    ap.add_argument("--output", default=str(ROOT / "outputs" / "v2-generalization" / "V2_RESULTS.md"))
    ap.add_argument("--json-output", default=str(ROOT / "outputs" / "v2-generalization" / "V2_RESULTS.json"))
    args = ap.parse_args()

    cfg = load_v2_experiment_config(args.config)
    sc = cfg.success_criteria
    warnings = []

    # dense nDCG@10 per dataset: v1 (baselines/real_<ds>.json) and v2 (eval/dense_<ds>.json)
    dense = {}
    for ds in DATASETS:
        v1 = _load(pathlib.Path(args.v1_dir) / "baselines" / f"real_{ds}.json")
        v2 = _load(pathlib.Path(args.v2_dir) / "eval" / f"dense_{ds}.json")
        if v1 is None:
            warnings.append(f"missing v1 dense report: real_{ds}.json")
        if v2 is None:
            warnings.append(f"missing v2 dense report: eval/dense_{ds}.json")
        dense[ds] = {"v1": _ndcg_by_model(v1), "v2": _ndcg_by_model(v2)}

    def best_student(by_model):
        cands = {k: v for k, v in by_model.items() if "boldt" in k.lower()}
        return max(cands.values()) if cands else None

    def metric(ds):  # prefer v2 student, else v1 student
        return best_student(dense[ds]["v2"]) or best_student(dense[ds]["v1"])

    # reranker promotion (v2 lift reports)
    rr_dt = _lift_delta(_load(pathlib.Path(args.v2_dir) / "reranker-lift-dt_test-v2.json"))
    rr_gq = _lift_delta(_load(pathlib.Path(args.v2_dir) / "reranker-lift-germanquad-v2.json"))
    # matryoshka retention (v1 or v2)
    mat = _load(pathlib.Path(args.v2_dir) / "real_matryoshka_germanquad.json") \
        or _load(pathlib.Path(args.v1_dir) / "baselines" / "real_matryoshka_germanquad.json")
    retention = None
    if mat and mat.get("matryoshka_sweep"):
        sweep = mat["matryoshka_sweep"]
        full, d256 = sweep.get("1024", {}).get("ndcg@10"), sweep.get("256", {}).get("ndcg@10")
        if full and d256:
            retention = round(d256 / full, 4)

    checks = {}

    def _chk(name, val, lo):
        ok = (val is not None) and (val >= lo)
        checks[name] = {"value": val, "min": lo, "pass": ok}
        return ok
    _chk("dense_germanquad_ndcg10", metric("germanquad"), sc.get("dense_germanquad_ndcg10_min"))
    _chk("dense_dt_test_ndcg10", metric("dt_test"), sc.get("dense_dt_test_ndcg10_min"))
    _chk("dense_gerdalir_ndcg10", metric("gerdalir"), sc.get("dense_gerdalir_ndcg10_min"))
    _chk("reranker_germanquad_delta", rr_gq, sc.get("reranker_germanquad_delta_min"))
    _chk("matryoshka_256_retention", retention, sc.get("matryoshka_256_retention_min"))
    rr_gate = (rr_dt is not None and rr_dt >= 0.0) and (rr_gq is not None and rr_gq >= 0.0)

    n_pass = sum(1 for c in checks.values() if c["pass"])
    verdict = ("improved" if n_pass == len(checks) and rr_gate
               else "failed" if not checks["dense_germanquad_ndcg10"]["pass"]
               and not checks["dense_dt_test_ndcg10"]["pass"] else "mixed")

    recs = []
    for name, c in checks.items():
        if not c["pass"]:
            recs.append(f"{name}: {c['value']} < target {c['min']} — scale/diversify v2 data or retrain.")
    if rr_dt is not None and not rr_gate:
        recs.append("reranker promotion gate FAILED (degrades a held-out set) — do not promote; "
                    "train on more diverse candidate lists.")
    if not recs:
        recs.append("All measured criteria met — promote candidate students and proceed to broader eval.")

    result = {"verdict": verdict, "criteria_passed": f"{n_pass}/{len(checks)}",
              "reranker_promotion_gate": "pass" if rr_gate else ("fail" if rr_dt is not None else "n/a"),
              "dense": dense, "reranker_delta": {"dt_test": rr_dt, "germanquad": rr_gq},
              "matryoshka_256_retention": retention, "checks": checks,
              "recommendations": recs, "warnings": warnings}

    lines = [f"# v2 results — verdict: **{verdict.upper()}** ({n_pass}/{len(checks)} criteria)", "",
             f"Reranker promotion gate: **{result['reranker_promotion_gate']}**", "",
             "## Dense retrieval nDCG@10 (best student per dataset)", "",
             "| dataset | v1 | v2 | min |", "|---|---:|---:|---:|"]
    for ds in DATASETS:
        lines.append(f"| {ds} | {best_student(dense[ds]['v1'])} | {best_student(dense[ds]['v2'])} | "
                     f"{sc.get('dense_' + ds.replace('-', '_') + '_ndcg10_min', '—')} |")
    lines += ["", "## Reranker lift (delta over first stage)",
              f"- DT-test: {rr_dt}", f"- GermanQuAD: {rr_gq} (target {sc.get('reranker_germanquad_delta_target')})",
              "", f"## Matryoshka 256-d retention: {retention} (min {sc.get('matryoshka_256_retention_min')})",
              "", "## Recommendations"]
    lines += [f"- {r}" for r in recs]
    if warnings:
        lines += ["", "## Warnings (missing inputs)"] + [f"- {w}" for w in warnings]
    md = "\n".join(lines) + "\n"

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(md, encoding="utf-8")
    pathlib.Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.json_output).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(f"verdict={verdict} criteria={n_pass}/{len(checks)} reranker_gate={result['reranker_promotion_gate']}")
    print(f"saved: {args.output}, {args.json_output}")
    if warnings:
        print(f"[warnings] {len(warnings)} missing input(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
