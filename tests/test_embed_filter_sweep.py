"""Tests for the v7 EmbedFilter sweep harness (stdlib core: planning, name-map, delta math)."""
import contextlib
import importlib.util
import io
import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SW = _load("eval_embed_filter_sweep")


class ResolveTests(unittest.TestCase):
    def test_name_map_and_skip_missing(self):
        ev_sets = {"webfaq": ("a", "b", "c", "primary"), "germanquad": ("x", "y", "z", "guardrail")}
        runnable, skipped = SW.resolve_eval_sets(
            ["webfaq_heldout", "local_rag", "germanquad"], ev_sets)
        # local_rag has no mapping/eval set -> skipped; webfaq/germanquad map but files absent here
        skipped_names = {s["name"] for s in skipped}
        self.assertIn("local_rag", skipped_names)
        # webfaq_heldout maps to "webfaq" but the dummy paths don't exist -> also skipped
        self.assertIn("webfaq_heldout", skipped_names)


class DeltaTests(unittest.TestCase):
    def test_deltas_vs_full_and_prefix(self):
        rows = [
            {"method": "full", "dim": 1024, "tau": None, "eval_set": "webfaq",
             "ndcg@10": 0.70, "recall@100": 0.97},
            {"method": "prefix", "dim": 256, "tau": None, "eval_set": "webfaq",
             "ndcg@10": 0.66, "recall@100": 0.95},
            {"method": "embedfilter", "dim": 256, "tau": 4, "eval_set": "webfaq",
             "ndcg@10": 0.68, "recall@100": 0.96},
        ]
        out = SW.compute_deltas(rows)
        ef = [r for r in out if r["method"] == "embedfilter"][0]
        self.assertEqual(ef["dNDCG10_vs_full"], round(0.68 - 0.70, 4))
        self.assertEqual(ef["dNDCG10_vs_prefix"], round(0.68 - 0.66, 4))   # beats prefix-256
        self.assertEqual(ef["dRecall100_vs_prefix"], round(0.96 - 0.95, 4))


class PlanTests(unittest.TestCase):
    def test_dry_run_plan_runs_no_ml(self):
        config = json.loads(
            (ROOT / "configs/experiments/v7_embedfilter.json").read_text(encoding="utf-8"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = SW.main(["--config", str(ROOT / "configs/experiments/v7_embedfilter.json"),
                          "--dry-run"])
        self.assertEqual(rc, 0)
        plan = json.loads(buf.getvalue())
        self.assertEqual(plan["status"], "dry_run")
        # local_rag is not on disk -> must appear as skipped, never fabricated
        self.assertIn("local_rag", {s["name"] for s in plan["eval_sets_skipped"]})

    def test_dry_run_imports_no_ml(self):
        code = (
            "import sys, importlib.util;"
            "spec=importlib.util.spec_from_file_location('s','scripts/eval_embed_filter_sweep.py');"
            "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
            "rc=m.main(['--config','configs/experiments/v7_embedfilter.json','--dry-run']);"
            "assert rc==0, rc;"
            "assert 'torch' not in sys.modules and 'sentence_transformers' not in sys.modules"
        )
        r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
