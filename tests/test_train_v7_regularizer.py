"""Tests for the opt-in v7 edge-spectrum regularizer (default OFF): plan + config + penalty math."""
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import train_modern as TM  # noqa: E402
from boldt_embed.config_teacher import load_student_training_config  # noqa: E402


class PlanTests(unittest.TestCase):
    def test_disabled_by_default(self):
        p = TM.plan_edge_spectrum_regularizer(None)
        self.assertFalse(p["active"])
        self.assertEqual(p["status"], "disabled")

    def test_enabled_zero_lambda_is_inactive(self):
        p = TM.plan_edge_spectrum_regularizer({"enabled": True, "lambda": 0.0})
        self.assertFalse(p["active"])
        self.assertEqual(p["status"], "enabled_zero_lambda")

    def test_active_when_enabled_and_lambda_positive(self):
        p = TM.plan_edge_spectrum_regularizer(
            {"enabled": True, "lambda": 0.1, "embed_filter_artifact": "outputs/embedfilter/x"})
        self.assertTrue(p["active"])
        self.assertEqual(p["embed_filter_artifact"], "outputs/embedfilter/x")


class ConfigTests(unittest.TestCase):
    def test_v7_training_config_loads_and_is_default_off(self):
        cfg = load_student_training_config(
            str(ROOT / "configs/experiments/v7_embedfilter_training.json"))
        reg = cfg.raw.get("edge_spectrum_regularizer")
        self.assertIsNotNone(reg)
        self.assertFalse(TM.plan_edge_spectrum_regularizer(reg)["active"])  # default OFF


class PenaltyMathTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch not available")

    def test_residual_outside_bulk_subspace(self):
        import torch
        basis = torch.eye(4)[:, :2]                       # keep dims 0,1
        kept = torch.tensor([[1.0, 2.0, 0.0, 0.0], [3.0, 1.0, 0.0, 0.0]])
        self.assertAlmostEqual(TM.edge_spectrum_penalty(kept, basis).item(), 0.0, places=5)
        edge = torch.tensor([[0.0, 0.0, 3.0, 4.0]])       # energy only in dropped dims
        self.assertAlmostEqual(TM.edge_spectrum_penalty(edge, basis).item(), 25.0, places=4)


class DryRunTests(unittest.TestCase):
    def test_train_dry_run_reports_regularizer_no_ml(self):
        cmd = [sys.executable, "scripts/train_modern_embedder.py", "--dry-run",
               "--student-config", "configs/experiments/v7_embedfilter_training.json"]
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("edge-spectrum-regularizer", r.stdout)


if __name__ == "__main__":
    unittest.main()
