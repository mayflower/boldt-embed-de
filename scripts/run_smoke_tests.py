#!/usr/bin/env python3
"""Curated deterministic CPU smoke tests (pure stdlib, no weights).

Exercises the full stack end-to-end quickly: structure, configs, data, hard negatives,
numeric core, the three dry-run trainers, and benchmark plumbing.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import config as cfgmod  # noqa: E402
from boldt_embed import data as datamod  # noqa: E402
from boldt_embed import eval_harness as eh  # noqa: E402
from boldt_embed import hard_negatives as hn  # noqa: E402
from boldt_embed import losses, matryoshka, merging, pooling  # noqa: E402
from boldt_embed.model_bidirectional import BidirectionalEmbedder  # noqa: E402
from boldt_embed.model_causal import CausalEmbedder  # noqa: E402
from boldt_embed.reranker import Reranker  # noqa: E402

CONFIGS = ROOT / "configs"
SAMPLES = ROOT / "data" / "samples"
BENCH = ROOT / "benchmarks" / "toy_de_retrieval.json"


def _check(name, fn):
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "passed": False, "detail": f"exception: {exc}"}
    return {"name": name, "passed": bool(ok), "detail": detail}


def c_structure():
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_repo.py"), "--format", "json"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return r.returncode == 0, "validate_repo pass" if r.returncode == 0 else r.stdout[-300:]


def c_configs():
    cfgmod.load_causal_config(CONFIGS / "training_causal.json")
    cfgmod.load_bidirectional_config(CONFIGS / "training_bidirectional.json")
    cfgmod.load_reranker_config(CONFIGS / "training_reranker.json")
    cfgmod.load_evaluation_config(CONFIGS / "evaluation.json")
    return True, "all 4 configs load"


def c_data():
    triples = datamod.load_jsonl(SAMPLES / "toy_triples_de.jsonl")
    pairs = datamod.load_jsonl(SAMPLES / "toy_pairs_de.jsonl")
    rep_t = datamod.validate_dataset(triples)
    rep_p = datamod.validate_dataset(pairs)
    ok = rep_t.ok and rep_p.ok and not datamod.check_licenses(triples + pairs)
    return ok, f"triples={rep_t.num_records} pairs={rep_p.num_records} errors={rep_t.errors + rep_p.errors}"


def c_hard_negs():
    samples = {
        "compound": "Die Kündigungsfrist beträgt drei Monate.",
        "negation": "Bei diesem Vertrag besteht ein Widerrufsrecht.",
        "legal_ref": "Gemäß § 543 BGB ist eine Kündigung möglich.",
        "dates_numbers": "Die Frist beträgt 14 Tage.",
        "regional_variant": "Im Jänner ist es kalt.",
        "entity_disambiguation": "Der VW Golf ist ein Auto von Volkswagen.",
    }
    missing = [cat for cat, txt in samples.items() if hn.GENERATORS[cat](txt) is None]
    return not missing, "all families generate" if not missing else f"missing: {missing}"


def c_numeric_core():
    assert pooling.l2_normalize([3.0, 4.0]) == [0.6, 0.8]
    assert pooling.mean_pool([[1.0, 0.0], [0.0, 1.0]], [1, 1]) == [0.5, 0.5]
    mid = merging.slerp([1.0, 0.0], [0.0, 1.0], 0.5)
    assert abs((mid[0] ** 2 + mid[1] ** 2) - 1.0) < 1e-9
    v = matryoshka.truncate_normalized([3.0, 4.0, 0.0], 2)
    assert abs(sum(x * x for x in v) - 1.0) < 1e-9
    good = losses.info_nce_loss([[1.0, 0.0]], [[1.0, 0.0]])
    assert good < 1e-3
    return True, "pooling/matryoshka/merging/losses OK"


def c_dry_runs():
    causal = CausalEmbedder.from_config(CONFIGS / "training_causal.json").dry_run(["q"], ["d"])
    bi = BidirectionalEmbedder.from_config(CONFIGS / "training_bidirectional.json").dry_run(["t"])
    rr = Reranker.from_config(CONFIGS / "training_reranker.json").dry_run("q", ["d"])
    ok = causal["status"] == "pass" and bi["status"] == "pass" and rr["status"] == "pass"
    return ok, "causal/bi/reranker dry-runs pass"


def c_benchmark():
    data = json.loads(BENCH.read_text(encoding="utf-8"))
    agg = eh.evaluate_bm25(data, (1, 10))["aggregate"]
    return agg["recall@1"] == 1.0, f"bm25 recall@1={agg['recall@1']}"


def run() -> dict:
    checks = [
        _check("structure", c_structure),
        _check("configs", c_configs),
        _check("data", c_data),
        _check("hard_negatives", c_hard_negs),
        _check("numeric_core", c_numeric_core),
        _check("dry_runs", c_dry_runs),
        _check("benchmark_plumbing", c_benchmark),
    ]
    return {"status": "pass" if all(c["passed"] for c in checks) else "fail", "checks": checks}


def render_markdown(report: dict) -> str:
    lines = ["# Repo Smoke Test Report", "", f"Status: **{report['status']}**", ""]
    for c in report["checks"]:
        lines.append(f"## {c['name']}")
        lines.append("PASS" if c["passed"] else "FAIL")
        lines.append(f"- {c['detail']}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    args = parser.parse_args()
    report = run()
    print(render_markdown(report) if args.format == "markdown"
          else json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
