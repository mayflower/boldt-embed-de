import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed.model_bidirectional import BidirectionalEmbedder  # noqa: E402

CONFIG = ROOT / "configs" / "training_bidirectional.json"


class TestBidirectionalDryRun(unittest.TestCase):
    def setUp(self):
        self.emb = BidirectionalEmbedder.from_config(CONFIG)

    def test_dry_run_report(self):
        report = self.emb.dry_run(["Ein deutscher Satz.", "Noch ein Satz."])
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["adaptation"], "masked_next_token_prediction")
        self.assertIn("mean", report["pooling_ablation"])
        self.assertEqual(report["num_texts"], 2)

    def test_mntp_plan(self):
        plan = self.emb.mntp_plan()
        self.assertEqual(plan["objective"], "masked_next_token_prediction")
        self.assertEqual(plan["enables"], "bidirectional_attention")


if __name__ == "__main__":
    unittest.main()
