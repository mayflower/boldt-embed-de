"""Stdlib tests for v2 benchmark task groups + eval-leakage guard. No network, no ML."""
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_baseline_benchmarks as RB  # noqa: E402
from boldt_embed import source_manifest as sm  # noqa: E402

TASKS = ROOT / "benchmarks" / "mteb_german_tasks.json"
MANIFEST = ROOT / "configs" / "data_sources_v2.json"
SCRIPT = ROOT / "scripts" / "run_baseline_benchmarks.py"


def _entry(sid, allowed):
    return sm.SourceEntry(source_id=sid, display_name=sid, source_type="hf_dataset",
                          domain="qa_wiki", license="X", allowed_for_training=allowed,
                          public_benchmark=True, eval_only=True, notes="", loader={})


class TestBenchmarkConfig(unittest.TestCase):
    def test_shipped_task_groups_valid(self):
        tg = RB.load_benchmark_tasks(TASKS)
        self.assertIn("retrieval_core", tg)
        self.assertEqual(RB.validate_benchmark_tasks(tg), [])

    def test_eval_only_false_flagged(self):
        tg = {"g": [{"name": "T", "eval_only": False, "allowed_for_training": False,
                     "metric_primary": "ndcg_at_10"}]}
        self.assertTrue(any("eval_only" in e for e in RB.validate_benchmark_tasks(tg)))

    def test_missing_metric_flagged(self):
        tg = {"g": [{"name": "T", "eval_only": True, "allowed_for_training": False}]}
        self.assertTrue(any("metric_primary" in e for e in RB.validate_benchmark_tasks(tg)))

    def test_no_leakage_with_shipped_manifest(self):
        tg = RB.load_benchmark_tasks(TASKS)
        entries = sm.load_source_manifest(MANIFEST)
        self.assertEqual(RB.check_eval_leakage_against_manifest(tg, entries), [])

    def test_leakage_detected_when_eval_set_trainable(self):
        tg = RB.load_benchmark_tasks(TASKS)
        entries = [_entry("germanquad", True)]   # eval set marked training-allowed
        bad = RB.check_eval_leakage_against_manifest(tg, entries)
        self.assertTrue(any("germanquad" in b for b in bad))


class TestCLI(unittest.TestCase):
    def test_task_group_filter_dry_run(self):
        out = subprocess.run([sys.executable, str(SCRIPT), "--tasks", str(TASKS),
                              "--task-group", "retrieval_core", "--dry-run"],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("group retrieval_core", out.stdout)
        self.assertNotIn("semantic_similarity", out.stdout)
        self.assertIn("dry-run-ok", out.stdout)

    def test_unknown_task_group_fails(self):
        out = subprocess.run([sys.executable, str(SCRIPT), "--tasks", str(TASKS),
                              "--task-group", "nope", "--dry-run"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 2)
        self.assertIn("unknown task-group", out.stderr)


if __name__ == "__main__":
    unittest.main()
