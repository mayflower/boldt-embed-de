import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import config  # noqa: E402

CONFIGS = ROOT / "configs"


class TestLoadShippedConfigs(unittest.TestCase):
    def test_causal(self):
        cfg = config.load_causal_config(CONFIGS / "training_causal.json")
        self.assertEqual(cfg.variant, "causal")
        self.assertEqual(cfg.embedding_dim, 1024)
        self.assertEqual(cfg.matryoshka_dims[0], 1024)
        self.assertIn("{query}", cfg.query_instruction)

    def test_bidirectional(self):
        cfg = config.load_bidirectional_config(CONFIGS / "training_bidirectional.json")
        self.assertEqual(cfg.variant, "bidirectional")
        self.assertEqual(cfg.adaptation, "masked_next_token_prediction")
        self.assertTrue(cfg.pooling_ablation)
        self.assertEqual(cfg.mntp_steps_dry_run, 2)  # read from 'mmtp_steps_dry_run'

    def test_reranker(self):
        cfg = config.load_reranker_config(CONFIGS / "training_reranker.json")
        self.assertEqual(cfg.variant, "reranker")
        self.assertIn("{query}", cfg.input_template)
        self.assertIn("{document}", cfg.input_template)

    def test_evaluation(self):
        cfg = config.load_evaluation_config(CONFIGS / "evaluation.json")
        self.assertIn("ndcg_at_10", cfg.metrics)
        self.assertTrue(cfg.report_metadata_required)


class TestValidation(unittest.TestCase):
    def test_ascending_matryoshka_rejected(self):
        errs = config.validate_config_dict(
            {"variant": "causal", "model_name_or_path": "m", "pooling": "eos",
             "embedding_dim": 128, "matryoshka_dims": [64, 128], "temperature": 0.05}
        )
        self.assertTrue(any("strictly decreasing" in e for e in errs), errs)

    def test_bad_temperature_rejected(self):
        errs = config.validate_config_dict(
            {"variant": "causal", "model_name_or_path": "m", "pooling": "eos",
             "embedding_dim": 64, "matryoshka_dims": [64], "temperature": 0.0}
        )
        self.assertTrue(any("temperature" in e for e in errs), errs)

    def test_unknown_pooling_rejected(self):
        errs = config.validate_config_dict(
            {"variant": "causal", "model_name_or_path": "m", "pooling": "banana",
             "embedding_dim": 64, "matryoshka_dims": [64], "temperature": 0.05}
        )
        self.assertTrue(any("pooling" in e for e in errs), errs)

    def test_reranker_template_requires_placeholders(self):
        errs = config.validate_config_dict(
            {"variant": "reranker", "model_name_or_path": "m",
             "input_template": "nur query: {query}", "positive_label": "Ja",
             "negative_label": "Nein", "max_length": 512}
        )
        self.assertTrue(any("{document}" in e for e in errs), errs)

    def test_evaluation_requires_metrics(self):
        errs = config.validate_config_dict({"metrics": [], "matryoshka_dims": [64],
                                            "report_metadata_required": ["commit"]})
        self.assertTrue(any("metrics" in e for e in errs), errs)

    def test_load_raises_on_invalid(self):
        with self.assertRaises(ValueError):
            config.load_causal_config(CONFIGS / "evaluation.json")  # wrong shape


if __name__ == "__main__":
    unittest.main()
