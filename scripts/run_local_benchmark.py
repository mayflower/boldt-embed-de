#!/usr/bin/env python3
"""Tiny deterministic German retrieval benchmark (pure stdlib).

Validates the metric + Matryoshka plumbing using a BM25 baseline and a deterministic
char-n-gram HashingEncoder stand-in. It is NOT a quality claim about the Boldt model.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import eval_harness as eh  # noqa: E402
from boldt_embed import data as datamod  # noqa: E402

BENCH = ROOT / "benchmarks" / "toy_de_retrieval.json"
STRESS = ROOT / "benchmarks" / "stress_cases_de.jsonl"
KS = (1, 3, 5, 10)
MATRYOSHKA = (256, 128, 64)
DISCLAIMER = (
    "This local benchmark validates metric/Matryoshka plumbing only with a BM25 baseline "
    "and a deterministic hashing stand-in. It is NOT evidence of Boldt embedding quality."
)


def run() -> dict:
    data = json.loads(BENCH.read_text(encoding="utf-8"))
    stress = datamod.load_jsonl(STRESS) if STRESS.exists() else []
    bm25 = eh.evaluate_bm25(data, KS)
    hashing = eh.evaluate_hashing(data, KS, matryoshka_dims=MATRYOSHKA, dim=256)
    return {
        "status": "pass",
        "benchmark": BENCH.name,
        "note": data.get("description"),
        "disclaimer": DISCLAIMER,
        "methods": {
            "bm25": {"aggregate": bm25["aggregate"], "queries": bm25["queries"]},
            "hashing_stand_in": {
                "aggregate": hashing["full"]["aggregate"],
                "matryoshka_by_dim": hashing.get("by_dim", {}),
            },
        },
        "stress_cases": eh.summarize_stress(stress),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Local German Retrieval Benchmark Report", "",
        f"Status: **{report['status']}**", f"Benchmark: `{report['benchmark']}`", "",
        f"> {report['disclaimer']}", "",
        "## BM25 baseline — aggregate", "", "| Metric | Value |", "|---|---:|",
    ]
    for k, v in sorted(report["methods"]["bm25"]["aggregate"].items()):
        lines.append(f"| {k} | {v:.4f} |")
    lines += ["", "## Hashing stand-in — aggregate", "", "| Metric | Value |", "|---|---:|"]
    for k, v in sorted(report["methods"]["hashing_stand_in"]["aggregate"].items()):
        lines.append(f"| {k} | {v:.4f} |")
    by_dim = report["methods"]["hashing_stand_in"]["matryoshka_by_dim"]
    if by_dim:
        lines += ["", "## Hashing stand-in — Matryoshka (nDCG@10 by dim)", "",
                  "| Dim | nDCG@10 | Recall@5 |", "|---:|---:|---:|"]
        for d in sorted(by_dim, key=int, reverse=True):
            agg = by_dim[d]
            lines.append(f"| {d} | {agg.get('ndcg@10', 0):.4f} | {agg.get('recall@5', 0):.4f} |")
    lines += ["", "## Stress-case coverage", ""]
    for case, n in report["stress_cases"].items():
        lines.append(f"- {case}: {n}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--save", action="store_true", help="write JSON+MD to outputs/benchmarks/")
    args = parser.parse_args()
    report = run()
    if args.save:
        out = ROOT / "outputs" / "benchmarks"
        out.mkdir(parents=True, exist_ok=True)
        (out / "local-benchmark-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "local-benchmark-report.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report) if args.format == "markdown" else json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
