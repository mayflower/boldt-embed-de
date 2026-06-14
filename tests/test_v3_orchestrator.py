"""Tests for the v3 real-domain orchestrator. Safe-by-default: dry-run plans only (no torch),
full needs the explicit flag, and any stage failure propagates to STATUS.json."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v3_real_domain_experiment.py"


def _run(work, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--work-dir", str(work), *extra],
        capture_output=True, text=True, cwd=str(ROOT))


class TestDryRun(unittest.TestCase):
    def test_dry_run_deterministic_command_list(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            r1 = _run(a, "--mode", "dry-run", "--target-count", "1000",
                      "--train-causal", "--train-reranker", "--eval")
            r2 = _run(b, "--mode", "dry-run", "--target-count", "1000",
                      "--train-causal", "--train-reranker", "--eval")
            self.assertEqual(r1.returncode, 0, r1.stderr)
            # stage sequence (from STATUS.json) is identical run-to-run
            s1 = [s["name"] for s in json.loads((pathlib.Path(a) / "STATUS.json").read_text())["stages"]]
            s2 = [s["name"] for s in json.loads((pathlib.Path(b) / "STATUS.json").read_text())["stages"]]
            self.assertEqual(s1, s2)
            self.assertEqual(s1[0], "acquire_sources")
            self.assertEqual(s1[-1], "release_gate")
            self.assertIn("domain_quality_gate", s1)
            self.assertIn("reranker_promotion_gate", s1)

    def test_dry_run_no_ml_and_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            r = _run(d, "--mode", "dry-run", "--target-count", "1000")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("dry-run-ok", r.stdout)
            for f in ("COMMANDS.md", "STATUS.json", "V3_RESULTS.json", "V3_RESULTS.md"):
                self.assertTrue((pathlib.Path(d) / f).exists(), f)
            self.assertEqual(json.loads((pathlib.Path(d) / "V3_RESULTS.json").read_text())["verdict"],
                             "planned")

    def test_dry_run_does_not_import_torch(self):
        # the orchestrator asserts no torch in dry-run; a clean exit proves it. Double-check via
        # a child that imports the module and inspects sys.modules.
        code = ("import sys; sys.path.insert(0, %r); sys.path.insert(0, %r);"
                "import run_v3_real_domain_experiment;"
                "assert 'torch' not in sys.modules; print('clean')") % (
                    str(ROOT / "scripts"), str(ROOT / "src"))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


class TestSafety(unittest.TestCase):
    def test_full_mode_blocked_without_flag(self):
        with tempfile.TemporaryDirectory() as d:
            r = _run(d, "--mode", "full", "--target-count", "1000")
            self.assertEqual(r.returncode, 2)
            self.assertIn("i-understand-this-runs-gpu", r.stderr)

    def test_failure_propagates_to_status(self):
        # smoke runs CPU stages against the shipped (placeholder) manifest: there is no real v3
        # data, so an early CPU stage fails — and the failure must be recorded + propagate.
        with tempfile.TemporaryDirectory() as d:
            r = _run(d, "--mode", "smoke", "--target-count", "1000")
            self.assertNotEqual(r.returncode, 0)
            status = json.loads((pathlib.Path(d) / "STATUS.json").read_text())
            stages = status["stages"]
            self.assertTrue(any(s["status"].startswith("failed") for s in stages),
                            [s["status"] for s in stages])
            # every stage after the first failure is skipped (prior failure)
            first_fail = next(i for i, s in enumerate(stages) if s["status"].startswith("failed"))
            for s in stages[first_fail + 1:]:
                self.assertIn("skipped", s["status"], s)
            self.assertEqual(status.get("verdict"), "failed")
            self.assertEqual(json.loads((pathlib.Path(d) / "V3_RESULTS.json").read_text())["verdict"],
                             "failed")


if __name__ == "__main__":
    unittest.main()
