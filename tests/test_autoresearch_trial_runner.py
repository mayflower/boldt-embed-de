"""Tests for the 20-minute trial runner (dry-run path; budget enforcement; crash capture)."""
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CONFIG = str(ROOT / "configs" / "autoresearch" / "experiments" / "current.json")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUN = _load("ar_run_trial")


def _metrics(out):
    return json.loads((pathlib.Path(out) / "metrics.json").read_text(encoding="utf-8"))


class BudgetTests(unittest.TestCase):
    def test_default_budget_is_20(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "run"
            rc = RUN.main(["--config", CONFIG, "--out", str(out), "--dry-run"])
            self.assertEqual(rc, 0)
            self.assertEqual(_metrics(out)["budget_minutes"], 20)
            self.assertFalse(_metrics(out)["invalid_for_default_loop"])

    def test_budget_over_20_fails_without_override(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "run"
            rc = RUN.main(["--config", CONFIG, "--out", str(out),
                           "--budget-minutes", "21", "--dry-run"])
            self.assertEqual(rc, 2)
            self.assertFalse((out / "metrics.json").exists())

    def test_budget_over_20_with_override_is_invalid_for_default_loop(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "run"
            rc = RUN.main(["--config", CONFIG, "--out", str(out), "--budget-minutes", "21",
                           "--allow-longer-than-20", "--dry-run"])
            self.assertEqual(rc, 0)
            self.assertTrue(_metrics(out)["invalid_for_default_loop"])

    def test_real_requires_allow_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "run"
            rc = RUN.main(["--config", CONFIG, "--out", str(out), "--real"])
            self.assertEqual(rc, 2)


class OutputTests(unittest.TestCase):
    def test_dry_run_writes_required_files(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "run"
            RUN.main(["--config", CONFIG, "--out", str(out), "--dry-run"])
            for name in ("config.resolved.json", "command.txt", "env.json",
                         "metrics.json", "run_card.md", "git.diffstat", "git.status"):
                self.assertTrue((out / name).exists(), f"missing {name}")
            self.assertEqual(_metrics(out)["mode"], "dry_run")

    def test_crash_writes_error_and_failed_metrics(self):
        from boldt_embed import autoresearch_recipe as recipe
        original = recipe.run_dense_trial

        def boom(*a, **k):
            raise RuntimeError("forced failure")

        recipe.run_dense_trial = boom
        try:
            with tempfile.TemporaryDirectory() as d:
                out = pathlib.Path(d) / "run"
                rc = RUN.main(["--config", CONFIG, "--out", str(out), "--dry-run"])
                self.assertEqual(rc, 1)
                self.assertTrue((out / "error.json").exists())
                self.assertEqual(_metrics(out)["status"], "crash")
                err = json.loads((out / "error.json").read_text(encoding="utf-8"))
                self.assertEqual(err["type"], "RuntimeError")
        finally:
            recipe.run_dense_trial = original


if __name__ == "__main__":
    unittest.main()
