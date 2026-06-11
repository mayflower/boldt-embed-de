"""Stdlib tests for the v2 orchestrator (dry-run safety, validation, command planning). No ML."""
import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v2_generalization_experiment.py"
CFG = ROOT / "configs" / "experiments" / "v2_generalization.json"
MANIFEST = ROOT / "configs" / "data_sources_v2.json"


def _run(extra, workdir):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(CFG), "--manifest", str(MANIFEST),
         "--work-dir", workdir] + extra, capture_output=True, text=True)


class TestDryRun(unittest.TestCase):
    def test_dry_run_plans_and_writes(self):
        with tempfile.TemporaryDirectory() as d:
            out = _run(["--mode", "dry-run", "--train-causal", "--train-bi-mntp",
                        "--train-reranker", "--eval"], d)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("dry-run-ok", out.stdout)
            self.assertTrue((pathlib.Path(d) / "COMMANDS.md").exists())
            status = json.loads((pathlib.Path(d) / "STATUS.json").read_text())
            names = [s["name"] for s in status["stages"]]
            for n in ("build_candidates", "teacher_cache", "train_causal", "train_bi_mntp",
                      "train_reranker", "eval_dense_gerdalir"):
                self.assertIn(n, names)
            self.assertTrue(all(s["status"] == "planned" for s in status["stages"]))

    def test_dry_run_deterministic(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            _run(["--mode", "dry-run", "--train-causal"], d1)
            _run(["--mode", "dry-run", "--train-causal"], d2)
            a = (pathlib.Path(d1) / "COMMANDS.md").read_text().replace(d1, "W")
            b = (pathlib.Path(d2) / "COMMANDS.md").read_text().replace(d2, "W")
            self.assertEqual(a, b)


class TestGuards(unittest.TestCase):
    def test_full_requires_ack_flag(self):
        with tempfile.TemporaryDirectory() as d:
            out = _run(["--mode", "full"], d)
            self.assertEqual(out.returncode, 2)
            self.assertIn("requires --i-understand-this-runs-gpu", out.stderr)

    def test_eval_only_false_config_fails(self):
        bad = copy.deepcopy(json.loads(CFG.read_text()))
        bad["public_benchmarks_eval_only"] = False
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(bad, f); f.close()
        with tempfile.TemporaryDirectory() as d:
            out = subprocess.run(
                [sys.executable, str(SCRIPT), "--config", f.name, "--manifest", str(MANIFEST),
                 "--work-dir", d, "--mode", "dry-run"], capture_output=True, text=True)
            self.assertNotEqual(out.returncode, 0)


if __name__ == "__main__":
    unittest.main()
