"""Stdlib tests for the modern embedding trainer's dataset + loss-plan layer + dry-run."""
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import teacher as T  # noqa: E402
from boldt_embed import train_modern as TM  # noqa: E402
from boldt_embed.config_teacher import load_student_training_config  # noqa: E402

CACHE = ROOT / "tests" / "fixtures" / "teacher_cache_small.jsonl"
STUDENT_CFG = ROOT / "configs" / "student_training_2026.json"


class TestDatasetBuild(unittest.TestCase):
    def setUp(self):
        self.rows = T.read_teacher_cache_jsonl(CACHE)
        self.examples = TM.build_train_dataset_from_teacher_cache(self.rows)

    def test_one_example_per_query_with_positive(self):
        self.assertEqual(len(self.examples), 2)  # q1, q2
        q1 = next(e for e in self.examples if "Mietkaution" in e["query"])
        self.assertIn("Dreifache", q1["positive"])  # best positive chosen
        self.assertEqual(len(q1["negatives"]), 2)   # d2, d5

    def test_negatives_hardest_first(self):
        q1 = next(e for e in self.examples if "Mietkaution" in e["query"])
        # d2 (reranker 0.9) is harder than d5 (-2.0)
        self.assertIn("Mietvertrag", q1["negatives"][0])
        self.assertEqual(q1["neg_scores"], [0.9, -2.0])

    def test_metadata(self):
        meta = TM.dataset_metadata(self.examples)
        self.assertEqual(meta["num_examples"], 2)
        self.assertTrue(meta["has_teacher_scores"])
        self.assertEqual(meta["num_with_negatives"], 2)

    def test_skips_query_without_positive(self):
        rows = [{"query_id": "qx", "doc_id": "dx", "query": "x", "document": "y",
                 "positive": False, "reranker_score": 0.1}]
        self.assertEqual(TM.build_train_dataset_from_teacher_cache(rows), [])


class TestLossPlan(unittest.TestCase):
    def setUp(self):
        self.cfg = load_student_training_config(STUDENT_CFG)

    def test_plan_without_ml_imports(self):
        plan = TM.plan_loss_stack(self.cfg, has_teacher_scores=True)
        self.assertEqual(plan["base_contrastive"], "CachedMultipleNegativesRankingLoss")
        self.assertIn("MatryoshkaLoss", plan["wrapped"])
        self.assertIn("MarginMSELoss", plan["distillation"])  # cfg has margin_mse + scores
        self.assertEqual(plan["matryoshka_dims"][0], 1024)

    def test_guide_switches_base_to_gist(self):
        plan = TM.plan_loss_stack(self.cfg, has_teacher_scores=False, use_guide=True)
        self.assertEqual(plan["base_contrastive"], "CachedGISTEmbedLoss")
        self.assertEqual(plan["distillation"], [])  # no scores -> no distill

    def test_plan_does_not_import_ml(self):
        # Isolated subprocess: a sibling torch test may have imported ML libs into this
        # process already. The contract is that building the dataset + loss plan imports none.
        code = (
            "import sys; sys.path.insert(0, %r);"
            "from boldt_embed import teacher as T, train_modern as TM;"
            "from boldt_embed.config_teacher import load_student_training_config as L;"
            "ex = TM.build_train_dataset_from_teacher_cache(T.read_teacher_cache_jsonl(%r));"
            "TM.plan_loss_stack(L(%r), TM.dataset_metadata(ex)['has_teacher_scores']);"
            "assert 'torch' not in sys.modules and 'sentence_transformers' not in sys.modules;"
            "print('clean')"
        ) % (str(ROOT / "src"), str(CACHE), str(STUDENT_CFG))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


class TestDryRunCLI(unittest.TestCase):
    def test_dry_run_no_ml(self):
        out = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train_modern_embedder.py"),
             "--student-config", str(STUDENT_CFG), "--teacher-cache", str(CACHE), "--dry-run"],
            capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("dry-run-ok", out.stdout)
        self.assertIn("loss-stack", out.stdout)
        self.assertIn("MatryoshkaLoss", out.stdout)


if __name__ == "__main__":
    unittest.main()
