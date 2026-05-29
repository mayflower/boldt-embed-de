import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data as datamod  # noqa: E402
from boldt_embed import eval_harness as eh  # noqa: E402
from boldt_embed import metrics  # noqa: E402

B = ROOT / "benchmarks"


class TestNewMetrics(unittest.TestCase):
    def test_spearman_monotonic(self):
        self.assertAlmostEqual(metrics.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(metrics.spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_v_measure_perfect_and_random(self):
        self.assertAlmostEqual(metrics.v_measure(["a", "a", "b", "b"], [0, 0, 1, 1]), 1.0)
        # label-preserving permutation is still perfect
        self.assertAlmostEqual(metrics.v_measure(["a", "a", "b", "b"], [1, 1, 0, 0]), 1.0)

    def test_accuracy(self):
        self.assertEqual(metrics.accuracy(["a", "b", "c"], ["a", "x", "c"]), 2 / 3)


class TestEvalSuite(unittest.TestCase):
    def setUp(self):
        self.encode = eh.HashingEncoder(dim=256, ngram=3).encode

    def test_sts_in_range(self):
        r = eh.evaluate_sts(datamod.load_jsonl(B / "sts_de.jsonl"), self.encode)
        self.assertGreaterEqual(r["spearman"], -1.0)
        self.assertLessEqual(r["spearman"], 1.0)
        self.assertEqual(r["n"], 8)

    def test_classification(self):
        cls = datamod.load_jsonl(B / "classification_de.jsonl")
        r = eh.evaluate_classification([c for c in cls if c["split"] == "train"],
                                       [c for c in cls if c["split"] == "test"], self.encode)
        self.assertGreaterEqual(r["accuracy"], 0.0)
        self.assertEqual(r["n_classes"], 3)

    def test_clustering(self):
        r = eh.evaluate_clustering(datamod.load_jsonl(B / "clustering_de.jsonl"), self.encode, k=3)
        self.assertGreaterEqual(r["v_measure"], 0.0)
        self.assertLessEqual(r["v_measure"], 1.0)

    def test_crosslingual_and_rag(self):
        for name in ("crosslingual_deen.json", "rag_de.json"):
            data = json.loads((B / name).read_text("utf-8"))
            agg = eh.retrieval_with_encoder(data, self.encode)
            self.assertIn("ndcg@10", agg)

    def test_stress_by_case(self):
        data = json.loads((B / "stress_de.json").read_text("utf-8"))
        r = eh.evaluate_stress(data)
        for case in ("compound", "legal_ref", "negation", "regional", "orthography", "number_date"):
            self.assertIn(case, r["by_case"], case)
        # orthography ss/ß is handled by normalization -> should retrieve the positive at rank 1
        self.assertEqual(r["by_case"]["orthography"]["recall@1"], 1.0)

    def test_efficiency(self):
        eff = eh.efficiency_report(256, (256, 128, 64))
        self.assertEqual(eff["bytes_per_vector_full_fp32"], 1024)
        self.assertEqual(eff["by_dim"][64]["bytes"], 256)


if __name__ == "__main__":
    unittest.main()
