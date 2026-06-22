"""Tests for the v6.1 dense eval core + dense gate (stdlib, no ML)."""
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EV = _load("eval_v6_1_dense_top50")
G = _load("check_v6_1_dense_gate")


def _passing_metrics():
    return {
        "webfaq": {"recall@50": 0.91, "recall@100": 0.965, "missing_positive_rate": 0.035,
                   "ndcg@10": 0.68, "matryoshka_256_retention": 0.96},
        "germanquad": {"ndcg@10": 0.89},
        "dt_test": {"ndcg@10": 0.95},
    }


class EvalCoreTests(unittest.TestCase):
    def test_recall_and_ndcg(self):
        # 2 queries; q1 gold at rank0 (perfect), q2 gold at rank 60 (in@100 not @50)
        rankings = {"q1": ["g1"] + [f"d{i}" for i in range(199)],
                    "q2": [f"e{i}" for i in range(60)] + ["g2"] + [f"e{i}" for i in range(60, 199)]}
        qrels = {"q1": {"g1"}, "q2": {"g2"}}
        m = EV.eval_rankings(rankings, qrels)
        self.assertEqual(m["recall@50"], 0.5)        # only q1 within top-50
        self.assertEqual(m["recall@100"], 1.0)       # both within top-100
        self.assertEqual(m["missing_positive_rate"], 0.0)
        self.assertEqual(m["n_queries"], 2)
        self.assertGreater(m["ndcg@10"], 0.4)        # q1 contributes 1.0, q2 0.0 at @10

    def test_missing_positive(self):
        rankings = {"q1": ["x", "y"]}
        m = EV.eval_rankings(rankings, {"q1": {"g1"}})
        self.assertEqual(m["recall@100"], 0.0)
        self.assertEqual(m["missing_positive_rate"], 1.0)
        self.assertEqual(m["oracle_ndcg@10"], 0.0)


class DenseGateTests(unittest.TestCase):
    def test_passes_clean(self):
        g = G.dense_gate(_passing_metrics())
        self.assertEqual(g["status"], "pass", g["failing"])
        self.assertTrue(g["independent_of_reranker"])
        self.assertIn("CAN be recommended", g["recommendation"])

    def test_fails_recall_at_50(self):
        m = _passing_metrics(); m["webfaq"]["recall@50"] = 0.883   # the v6 shortfall
        g = G.dense_gate(m)
        self.assertEqual(g["status"], "fail")
        self.assertIn("webfaq_recall_at_50", g["failed_targets"])
        self.assertIn("Do NOT recommend", g["recommendation"])

    def test_fails_recall_at_100(self):
        m = _passing_metrics(); m["webfaq"]["recall@100"] = 0.94
        self.assertIn("webfaq_recall_at_100", G.dense_gate(m)["failed_targets"])

    def test_fails_guardrail_germanquad(self):
        m = _passing_metrics(); m["germanquad"]["ndcg@10"] = 0.80
        self.assertIn("germanquad_ndcg_at_10", G.dense_gate(m)["failed_targets"])

    def test_fails_dt_test(self):
        m = _passing_metrics(); m["dt_test"]["ndcg@10"] = 0.90
        self.assertIn("dt_test_ndcg_at_10", G.dense_gate(m)["failed_targets"])

    def test_fails_matryoshka_retention(self):
        m = _passing_metrics(); m["webfaq"]["matryoshka_256_retention"] = 0.90
        self.assertIn("matryoshka_256_retention", G.dense_gate(m)["failed_targets"])

    def test_fails_missing_rate(self):
        m = _passing_metrics(); m["webfaq"]["missing_positive_rate"] = 0.08
        self.assertIn("webfaq_missing_positive_rate", G.dense_gate(m)["failed_targets"])

    def test_leakage_fails(self):
        self.assertEqual(G.dense_gate(_passing_metrics(), public_eval_leakage=True)["status"], "fail")

    def test_gate_is_dense_only(self):
        # the gate never references reranker/policy — its checks are all dense retrieval metrics
        g = G.dense_gate(_passing_metrics())
        names = " ".join(c["check"] for c in g["checks"])
        for forbidden in ("reranker", "policy", "bounded", "abstain"):
            self.assertNotIn(forbidden, names)
        self.assertEqual(g["based_on"], "dense retrieval quality only")

    def test_extract_from_summary(self):
        summary = {"dense-v6.1": {"webfaq": {"recall@50": 0.91, "recall@100": 0.965,
                                             "missing_positive_rate": 0.03, "ndcg@10": 0.68,
                                             "matryoshka_256_retention": 0.96},
                                  "germanquad": {"ndcg@10": 0.89}, "dt_test": {"ndcg@10": 0.95}}}
        m = G._extract(summary, "dense-v6.1")
        self.assertEqual(m["webfaq"]["recall@50"], 0.91)
        self.assertEqual(G.dense_gate(m)["status"], "pass")


if __name__ == "__main__":
    unittest.main()
