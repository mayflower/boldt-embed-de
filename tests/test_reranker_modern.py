"""Stdlib tests for modern reranker example builders + lift metrics + dry-runs."""
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import reranker_modern as RM  # noqa: E402
from boldt_embed import teacher as T  # noqa: E402

CACHE = ROOT / "tests" / "fixtures" / "teacher_cache_small.jsonl"
CANDS = ROOT / "tests" / "fixtures" / "rerank_candidates.jsonl"
RERANK_CFG = ROOT / "configs" / "training_reranker.json"
TEACHER_CFG = ROOT / "configs" / "teacher_models.json"


class TestBuilders(unittest.TestCase):
    def setUp(self):
        self.rows = T.read_teacher_cache_jsonl(CACHE)

    def test_pointwise_labels(self):
        ex = RM.build_reranker_examples_from_teacher_cache(self.rows)
        labels = {e["document"][:10]: e["label"] for e in ex}
        self.assertTrue(any(v == 1.0 for v in labels.values()))
        self.assertTrue(any(v == 0.0 for v in labels.values()))

    def test_pointwise_teacher_labels_are_soft(self):
        ex = RM.build_reranker_examples_from_teacher_cache(self.rows, label_mode="teacher")
        self.assertTrue(all(0.0 <= e["label"] <= 1.0 for e in ex))
        self.assertTrue(any(0.0 < e["label"] < 1.0 for e in ex))

    def test_pairwise_from_list_rows(self):
        pairs = RM.build_pairwise_examples(self.rows)
        self.assertTrue(pairs)
        for p in pairs:
            self.assertIn("positive", p)
            self.assertIn("negative", p)
            self.assertNotEqual(p["positive"], p["negative"])

    def test_listwise_target_distribution(self):
        batches = RM.build_listwise_batches(self.rows)
        self.assertTrue(batches)
        for b in batches:
            self.assertEqual(len(b["documents"]), len(b["target"]))
            self.assertAlmostEqual(sum(b["target"]), 1.0, places=6)
        # q1: positive d1 has highest reranker score -> highest target prob
        q1 = next(b for b in batches if "Mietkaution" in b["query"])
        self.assertEqual(q1["target"].index(max(q1["target"])), q1["labels"].index(1.0))

    def test_softmax_temperature(self):
        sharp = RM.softmax([2.0, 1.0, 0.0], temperature=0.5)
        flat = RM.softmax([2.0, 1.0, 0.0], temperature=5.0)
        self.assertGreater(sharp[0], flat[0])  # lower temp -> sharper


class TestLiftMetrics(unittest.TestCase):
    def test_rerank_improves_over_first_stage(self):
        cids = ["d1", "d2", "d3"]
        pos = ["d2"]
        first = RM.first_stage_metrics(cids, pos, (10,))["ndcg@10"]
        rer = RM.rerank_metrics(cids, [0.1, 0.9, 0.2], pos, (10,))["ndcg@10"]
        self.assertGreater(rer, first)

    def test_oracle_is_perfect_when_positive_present(self):
        self.assertAlmostEqual(RM.oracle_metrics(["d1", "d2", "d3"], ["d2"], (10,))["ndcg@10"], 1.0)

    def test_positive_in_top_k(self):
        cids = ["d1", "d2", "d3"]
        self.assertEqual(RM.positive_in_top_k(cids, [0.1, 0.9, 0.2], ["d2"], 1), 1.0)
        self.assertEqual(RM.positive_in_top_k(cids, [0.9, 0.1, 0.2], ["d2"], 1), 0.0)

    def test_noop_scorer_reproduces_first_stage(self):
        cids = ["d1", "d2", "d3"]
        pos = ["d2"]
        first = RM.first_stage_metrics(cids, pos, (10,))["ndcg@10"]
        same = RM.rerank_metrics(cids, [3, 2, 1], pos, (10,))["ndcg@10"]  # already-desc scores
        self.assertEqual(first, same)


class TestDryRuns(unittest.TestCase):
    def test_train_reranker_dry_run(self):
        out = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train_modern_reranker.py"),
             "--config", str(RERANK_CFG), "--teacher-cache", str(CACHE),
             "--loss", "listwise", "--dry-run"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("dry-run-ok", out.stdout)
        self.assertIn("listwise_queries", out.stdout)

    def test_eval_lift_dry_run(self):
        out = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "eval_reranker_lift.py"),
             "--candidates", str(CANDS), "--config", str(RERANK_CFG), "--dry-run"],
            capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("dry-run-ok", out.stdout)
        self.assertIn("first_stage_ndcg@10", out.stdout)
        self.assertIn("oracle_ndcg@10", out.stdout)


if __name__ == "__main__":
    unittest.main()
