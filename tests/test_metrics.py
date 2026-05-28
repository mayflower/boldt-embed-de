import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import metrics  # noqa: E402


class TestMetrics(unittest.TestCase):
    def test_perfect_ranking(self):
        ranked = ["a", "b", "c"]
        pos = {"a"}
        self.assertEqual(metrics.recall_at_k(ranked, pos, 1), 1.0)
        self.assertEqual(metrics.mrr_at_k(ranked, pos, 10), 1.0)
        self.assertAlmostEqual(metrics.ndcg_at_k(ranked, pos, 10), 1.0)

    def test_positive_at_rank_three(self):
        ranked = ["a", "b", "c"]
        pos = {"c"}
        self.assertEqual(metrics.recall_at_k(ranked, pos, 1), 0.0)
        self.assertEqual(metrics.recall_at_k(ranked, pos, 3), 1.0)
        self.assertAlmostEqual(metrics.mrr_at_k(ranked, pos, 10), 1.0 / 3.0)
        # nDCG@10 for a single positive at rank 3 == 1/log2(4) == 0.5
        self.assertAlmostEqual(metrics.ndcg_at_k(ranked, pos, 10), 0.5)

    def test_map_two_positives(self):
        ranked = ["a", "x", "b"]
        pos = {"a", "b"}
        # AP = (1/1 + 2/3) / 2
        self.assertAlmostEqual(metrics.average_precision_at_k(ranked, pos, 10), (1.0 + 2.0 / 3.0) / 2)

    def test_no_positive_in_top_k(self):
        ranked = ["x", "y", "z"]
        pos = {"a"}
        self.assertEqual(metrics.mrr_at_k(ranked, pos, 3), 0.0)
        self.assertEqual(metrics.ndcg_at_k(ranked, pos, 3), 0.0)

    def test_aggregate(self):
        rows = [{"ndcg@10": 1.0}, {"ndcg@10": 0.0}]
        self.assertEqual(metrics.aggregate(rows)["ndcg@10"], 0.5)

    def test_metrics_for_query_keys(self):
        m = metrics.metrics_for_query(["a"], {"a"}, ks=(1, 10))
        self.assertIn("ndcg@1", m)
        self.assertIn("recall@10", m)


if __name__ == "__main__":
    unittest.main()
