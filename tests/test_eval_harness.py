import json
import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import eval_harness as eh  # noqa: E402

BENCH = ROOT / "benchmarks" / "toy_de_retrieval.json"


def load_bench():
    return json.loads(BENCH.read_text(encoding="utf-8"))


class TestBM25(unittest.TestCase):
    def test_bm25_perfect_on_toy(self):
        # The toy queries are lexically aligned with their positives -> BM25 is perfect.
        result = eh.evaluate_bm25(load_bench(), ks=(1, 3, 10))
        agg = result["aggregate"]
        self.assertAlmostEqual(agg["recall@1"], 1.0)
        self.assertAlmostEqual(agg["mrr@10"], 1.0)
        self.assertAlmostEqual(agg["ndcg@10"], 1.0)


class TestHashingEncoder(unittest.TestCase):
    def test_unit_norm_and_deterministic(self):
        enc = eh.HashingEncoder(dim=64, ngram=3)
        v1 = enc.encode(["Kündigungsfrist Mietwohnung"])[0]
        v2 = eh.HashingEncoder(dim=64, ngram=3).encode(["Kündigungsfrist Mietwohnung"])[0]
        self.assertEqual(v1, v2)  # deterministic across instances
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in v1)), 1.0)

    def test_evaluate_hashing_structure_and_ranges(self):
        result = eh.evaluate_hashing(load_bench(), ks=(1, 5, 10),
                                     matryoshka_dims=(256, 128, 64), dim=256)
        agg = result["full"]["aggregate"]
        for value in agg.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertEqual(set(result["by_dim"]), {256, 128, 64})

    def test_matryoshka_dims_above_encoder_dim_are_skipped(self):
        result = eh.evaluate_hashing(load_bench(), ks=(10,),
                                     matryoshka_dims=(1024, 64), dim=256)
        self.assertNotIn(1024, result["by_dim"])
        self.assertIn(64, result["by_dim"])


class TestStress(unittest.TestCase):
    def test_stress_summary(self):
        cases = [{"case": "compound"}, {"case": "compound"}, {"case": "negation"}]
        self.assertEqual(eh.summarize_stress(cases), {"compound": 2, "negation": 1})


if __name__ == "__main__":
    unittest.main()
