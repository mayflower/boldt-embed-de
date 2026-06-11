"""Stdlib tests for v2 reranker: candidate-list loss builders + promotion gate. No ML."""
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data_pipeline as dp  # noqa: E402
from boldt_embed import reranker_modern as RM  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
CL = FIX / "reranker_train_v2_tiny.jsonl"
GATE = ROOT / "scripts" / "check_reranker_promotion_gate.py"
TRAIN = ROOT / "scripts" / "train_modern_reranker.py"
RERANK_CFG = ROOT / "configs" / "training_reranker.json"


class TestCandidateListLosses(unittest.TestCase):
    def setUp(self):
        self.rows = list(dp.stream_jsonl(CL))

    def test_pointwise(self):
        ex = RM.candidate_lists_to_pointwise(self.rows)
        self.assertEqual(len(ex), 5)  # 3 + 2 candidates
        self.assertTrue(any(e["label"] == 1.0 for e in ex))
        self.assertTrue(any(e["label"] == 0.0 for e in ex))

    def test_pairwise(self):
        pairs = RM.candidate_lists_to_pairwise(self.rows)
        self.assertEqual(len(pairs), 3)  # q1: 1x2, q2: 1x1
        self.assertTrue(all(p["positive"] != p["negative"] for p in pairs))

    def test_listwise_target_from_teacher_scores(self):
        batches = RM.candidate_lists_to_listwise(self.rows)
        self.assertEqual(len(batches), 2)
        for b in batches:
            self.assertAlmostEqual(sum(b["target"]), 1.0, places=6)
        q1 = next(b for b in batches if "Mietkaution" in b["query"])
        self.assertEqual(q1["target"].index(max(q1["target"])), q1["labels"].index(1.0))

    def test_plan_mixed(self):
        comps = RM.plan_reranker_loss("mixed")["components"]
        self.assertIn("BCEWithLogitsLoss", comps)
        self.assertIn("MarginRankingLoss", comps)
        self.assertIn("KLDivLoss(listwise)", comps)


class TestPromotionGate(unittest.TestCase):
    def _run(self, dt, gq):
        return subprocess.run([sys.executable, str(GATE), "--dt-test", str(FIX / dt),
                               "--germanquad", str(FIX / gq)], capture_output=True, text=True)

    def test_fails_when_germanquad_drops(self):
        out = self._run("reranker_lift_dt_pass.json", "reranker_lift_germanquad_fail.json")
        self.assertEqual(out.returncode, 1)
        self.assertIn('"status": "fail"', out.stdout)
        self.assertIn("do NOT promote", out.stdout)

    def test_passes_when_neutral_or_lift(self):
        out = self._run("reranker_lift_dt_pass.json", "reranker_lift_germanquad_pass.json")
        self.assertEqual(out.returncode, 0)
        self.assertIn('"status": "pass"', out.stdout)


class TestTrainDryRun(unittest.TestCase):
    def test_candidate_lists_dry_run(self):
        out = subprocess.run(
            [sys.executable, str(TRAIN), "--config", str(RERANK_CFG),
             "--candidate-lists", str(CL), "--loss", "mixed", "--dry-run"],
            capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("candidate-lists", out.stdout)
        self.assertIn("loss_components", out.stdout)
        self.assertIn("dry-run-ok", out.stdout)


if __name__ == "__main__":
    unittest.main()
