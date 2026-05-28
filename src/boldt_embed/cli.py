"""Console entrypoints (source-checkout convenience wrappers).

These power the ``boldt-validate`` / ``boldt-smoke`` / ``boldt-bench`` console scripts
declared in pyproject. They assume a source checkout (data/benchmarks live in the repo).
The canonical entrypoints remain ``python scripts/*.py``.
"""
from __future__ import annotations

import json
import pathlib
from typing import Optional, Sequence

from . import config as cfgmod
from . import data as datamod
from . import eval_harness as eh

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main_validate(argv: Optional[Sequence[str]] = None) -> int:
    results = {}
    for name in ("training_causal", "training_bidirectional", "training_reranker", "evaluation"):
        d = json.loads((ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8"))
        results[name] = cfgmod.validate_config_dict(d)
    ok = all(not v for v in results.values())
    _emit({"status": "pass" if ok else "fail", "configs": results})
    return 0 if ok else 1


def main_bench(argv: Optional[Sequence[str]] = None) -> int:
    data = json.loads((ROOT / "benchmarks" / "toy_de_retrieval.json").read_text(encoding="utf-8"))
    res = eh.evaluate_bm25(data, (1, 5, 10))
    _emit({"status": "pass", "bm25_aggregate": res["aggregate"]})
    return 0


def main_smoke(argv: Optional[Sequence[str]] = None) -> int:
    checks = {}
    checks["configs_ok"] = main_validate() == 0
    triples = datamod.load_jsonl(ROOT / "data" / "samples" / "toy_triples_de.jsonl")
    checks["data_ok"] = datamod.validate_dataset(triples).ok
    data = json.loads((ROOT / "benchmarks" / "toy_de_retrieval.json").read_text(encoding="utf-8"))
    checks["bm25_perfect"] = eh.evaluate_bm25(data, (1,))["aggregate"]["recall@1"] == 1.0
    ok = all(checks.values())
    _emit({"status": "pass" if ok else "fail", "checks": checks})
    return 0 if ok else 1
