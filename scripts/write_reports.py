#!/usr/bin/env python3
"""Run every validation gate and persist reports under outputs/ (pure stdlib).

Writes:
  outputs/validation/validation-report.{json,md}
  outputs/test-runs/smoke-test-report.{json,md}
  outputs/benchmarks/local-benchmark-report.{json,md}   (via run_local_benchmark --save)
  outputs/test-runs/unittest-report.txt
  outputs/SUMMARY.md
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "outputs"


def _run(args, **kw):
    return subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, **kw)


def _save(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    py = sys.executable
    statuses = {}

    for name, script, out_sub, out_name in [
        ("validation", "validate_repo.py", "validation", "validation-report"),
        ("smoke", "run_smoke_tests.py", "test-runs", "smoke-test-report"),
    ]:
        j = _run([py, str(SCRIPTS / script), "--format", "json"])
        m = _run([py, str(SCRIPTS / script), "--format", "markdown"])
        _save(OUT / out_sub / f"{out_name}.json", j.stdout)
        _save(OUT / out_sub / f"{out_name}.md", m.stdout)
        try:
            statuses[name] = json.loads(j.stdout).get("status")
        except Exception:
            statuses[name] = "error"

    bench = _run([py, str(SCRIPTS / "run_local_benchmark.py"), "--format", "json", "--save"])
    try:
        statuses["benchmark_plumbing"] = json.loads(bench.stdout).get("status")
    except Exception:
        statuses["benchmark_plumbing"] = "error"

    unit = _run([py, "-m", "unittest", "discover", "-s", "tests"])
    _save(OUT / "test-runs" / "unittest-report.txt", unit.stdout)
    statuses["unittest"] = "pass" if unit.returncode == 0 else "fail"

    overall = "pass" if all(v == "pass" for v in statuses.values()) else "fail"
    summary = ["# Validation Summary", "", f"Overall: **{overall}**", "",
               "| Gate | Status |", "|---|---|"]
    summary += [f"| {k} | {v} |" for k, v in statuses.items()]
    summary += ["", "_Local benchmark validates plumbing only; not a model-quality claim._"]
    _save(OUT / "SUMMARY.md", "\n".join(summary))

    print("\n".join(summary))
    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
