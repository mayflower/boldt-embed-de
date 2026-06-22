"""Tests for the v6.1 dense-retriever config validator (stdlib, no ML)."""
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed.v6_1_dense_config import (  # noqa: E402
    load_v6_1_dense_config, validate_v6_1_dense_config, is_reranker_training_allowed)

CONFIG_PATH = ROOT / "configs" / "experiments" / "v6_1_dense_top50.json"


def _cfg():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


class RealConfigTests(unittest.TestCase):
    def test_real_config_is_valid(self):
        self.assertEqual(validate_v6_1_dense_config(_cfg()), [])

    def test_loader_returns_validated_config(self):
        cfg = load_v6_1_dense_config(CONFIG_PATH)
        self.assertEqual(cfg["experiment_id"], "v6.1-dense-top50")

    def test_real_config_is_dense_only(self):
        cfg = _cfg()
        self.assertFalse(cfg["reranker_training_enabled"])
        self.assertFalse(is_reranker_training_allowed(cfg))

    def test_training_mix_sums_to_one(self):
        self.assertAlmostEqual(sum(_cfg()["training_mix"].values()), 1.0, places=9)


class ValidationRuleTests(unittest.TestCase):
    def test_fractions_must_sum_to_one(self):
        c = _cfg(); c["training_mix"]["webfaq_real"] = 0.50   # now sums to 1.05
        errs = validate_v6_1_dense_config(c)
        self.assertTrue(any("sum to 1.0" in e for e in errs))

    def test_reranker_training_must_be_false(self):
        c = _cfg(); c["reranker_training_enabled"] = True
        self.assertTrue(any("reranker_training_enabled must be false" in e
                            for e in validate_v6_1_dense_config(c)))

    def test_reranker_training_missing_is_rejected(self):
        c = _cfg(); del c["reranker_training_enabled"]
        self.assertTrue(any("reranker_training_enabled must be false" in e
                            for e in validate_v6_1_dense_config(c)))

    def test_public_benchmarks_must_be_eval_only(self):
        c = _cfg(); c["public_benchmarks_eval_only"] = False
        self.assertTrue(any("public_benchmarks_eval_only must be true" in e
                            for e in validate_v6_1_dense_config(c)))

    def test_target_metrics_must_be_numeric(self):
        c = _cfg(); c["target_metrics"]["webfaq_recall_at_50_min"] = "0.90"
        self.assertTrue(any("must be numeric" in e for e in validate_v6_1_dense_config(c)))

    def test_missing_target_metric_flagged(self):
        c = _cfg(); del c["target_metrics"]["matryoshka_256_retention_min"]
        self.assertTrue(any("matryoshka_256_retention_min" in e
                            for e in validate_v6_1_dense_config(c)))

    def test_bool_is_not_a_valid_number(self):
        c = _cfg(); c["target_metrics"]["webfaq_ndcg_at_10_min"] = True
        self.assertTrue(any("must be numeric" in e for e in validate_v6_1_dense_config(c)))

    def test_hard_negative_sources_required(self):
        c = _cfg(); c["hard_negative_sources"] = []
        self.assertTrue(any("hard_negative_sources" in e for e in validate_v6_1_dense_config(c)))

    def test_negative_fraction_rejected(self):
        c = _cfg(); c["training_mix"]["local_rag"] = -0.05; c["training_mix"]["webfaq_real"] = 0.50
        self.assertTrue(any("non-negative" in e for e in validate_v6_1_dense_config(c)))

    def test_loader_fails_closed_on_invalid(self):
        bad = ROOT / "tests" / "fixtures" / "_v6_1_bad_tmp.json"
        c = _cfg(); c["reranker_training_enabled"] = True
        bad.write_text(json.dumps(c), encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                load_v6_1_dense_config(bad)
        finally:
            bad.unlink()


if __name__ == "__main__":
    unittest.main()
