#!/usr/bin/env python3
"""Project-level validation gate (prompt 10). Runs every available stdlib check and reports
pass/fail per check, never claiming completion unless all pass or failures are surfaced.

Checks: unit tests · config parsing · data schema validation · prompt/template linting ·
tiny benchmark run · training dry run · evaluation dry run · model-card completeness.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PY = sys.executable


def _ok(args) -> tuple[bool, str]:
    r = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    return r.returncode == 0, tail[:160]


def run() -> dict:
    checks = {}
    checks["unit_tests"] = _ok([PY, "-m", "unittest", "discover", "-s", "tests"])
    # config parsing + prompt/template linting + structure + model-card completeness:
    checks["structure_and_model_cards"] = _ok([PY, str(SCRIPTS / "validate_repo.py"), "--format", "json"])
    checks["config_parsing"] = _ok([PY, "-c",
        "import sys; sys.path.insert(0,'src'); from boldt_embed import config as c;"
        "[getattr(c,f)('configs/'+n) for f,n in "
        "[('load_causal_config','training_causal.json'),"
        "('load_bidirectional_config','training_bidirectional.json'),"
        "('load_reranker_config','training_reranker.json'),"
        "('load_evaluation_config','evaluation.json')]]; print('configs ok')"])
    checks["data_schema"] = _ok([PY, str(SCRIPTS / "validate_data_schema.py"), "--format", "json"])
    checks["tiny_benchmark"] = _ok([PY, str(SCRIPTS / "run_local_benchmark.py"), "--format", "json"])
    checks["training_dry_run"] = _ok([PY, str(SCRIPTS / "train_causal.py"), "--dry-run"])
    checks["training_dry_run_bi"] = _ok([PY, str(SCRIPTS / "train_bidirectional.py"), "--dry-run"])
    checks["training_dry_run_reranker"] = _ok([PY, str(SCRIPTS / "train_reranker.py"), "--dry-run"])
    checks["evaluation_dry_run"] = _ok([PY, str(SCRIPTS / "run_eval_suite.py"), "--format", "json"])
    status = "pass" if all(ok for ok, _ in checks.values()) else "fail"
    return {"status": status, "checks": {k: {"passed": ok, "detail": d} for k, (ok, d) in checks.items()}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    args = parser.parse_args()
    report = run()
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"# Project Validation\n\nStatus: **{report['status']}**\n")
        for name, c in report["checks"].items():
            print(f"- [{'x' if c['passed'] else ' '}] {name}: {c['detail']}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
