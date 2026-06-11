"""Stdlib tests for v2 student-training support (hard-neg dataset, flags, dry-run). No ML."""
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import train_modern as TM  # noqa: E402
from boldt_embed import teacher as T  # noqa: E402
from boldt_embed.config_teacher import load_student_training_config  # noqa: E402

SCRIPT = ROOT / "scripts" / "train_modern_embedder.py"
STUDENT_CFG = ROOT / "configs" / "student_training_2026.json"
HARDNEG = ROOT / "tests" / "fixtures" / "hardneg_v2_tiny.jsonl"
CACHE = ROOT / "tests" / "fixtures" / "teacher_cache_v2_tiny.jsonl"


class TestHardNegDataset(unittest.TestCase):
    def test_builds_triplets(self):
        rows = list(T.stream_jsonl(HARDNEG))
        ex = TM.build_train_dataset_from_hardneg(rows)
        self.assertEqual(len(ex), 2)
        q1 = next(e for e in ex if "Mietkaution" in e["query"])
        self.assertIn("Dreifache", q1["positive"])
        self.assertEqual(len(q1["negatives"]), 2)
        self.assertEqual(q1["neg_scores"], [0.4, -4.0])


class TestLossPlanFlags(unittest.TestCase):
    def setUp(self):
        self.cfg = load_student_training_config(STUDENT_CFG)

    def test_distillation_forced_off(self):
        plan = TM.plan_loss_stack(self.cfg, has_teacher_scores=True, use_distillation=False)
        self.assertEqual(plan["distillation"], [])
        self.assertFalse(plan["teacher_distillation_active"])

    def test_distillation_auto_on_with_scores(self):
        plan = TM.plan_loss_stack(self.cfg, has_teacher_scores=True, use_distillation=None)
        self.assertIn("MarginMSELoss", plan["distillation"])


class TestDryRunCLI(unittest.TestCase):
    def _run(self, extra):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--student-config", str(STUDENT_CFG)] + extra + ["--dry-run"],
            capture_output=True, text=True)

    def test_hardneg_dry_run(self):
        out = self._run(["--hard-negatives", str(HARDNEG), "--teacher-cache", str(CACHE)])
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("hard-negatives", out.stdout)
        self.assertIn("triplet examples", out.stdout)
        self.assertIn("dry-run-ok", out.stdout)

    def test_bidirectional_flag_recorded(self):
        out = self._run(["--teacher-cache", str(CACHE), "--bidirectional", "true"])
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("bidirectional=True", out.stdout)

    def test_distillation_off_flag(self):
        out = self._run(["--teacher-cache", str(CACHE), "--use-teacher-score-distillation", "false"])
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn('"teacher_distillation_active": false', out.stdout)


if __name__ == "__main__":
    unittest.main()
