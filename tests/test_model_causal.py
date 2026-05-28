import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed.model_causal import CausalEmbedder  # noqa: E402

CONFIG = ROOT / "configs" / "training_causal.json"


class TestCausalEmbedderDryRun(unittest.TestCase):
    def setUp(self):
        self.emb = CausalEmbedder.from_config(CONFIG)

    def test_build_inputs_formats_query(self):
        built = self.emb.build_inputs(["kündigungsfrist"], ["Ein Dokument über Mietrecht."])
        self.assertIn("kündigungsfrist", built["queries"][0])
        self.assertIn("Instruct:", built["queries"][0])
        self.assertEqual(built["documents"][0], "Ein Dokument über Mietrecht.")

    def test_dry_run_report(self):
        report = self.emb.dry_run(["q1", "q2"], ["d1", "d2"])
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["num_queries"], 2)
        self.assertEqual(report["embedding_dim"], 1024)
        self.assertEqual(report["matryoshka_dims"][0], 1024)
        self.assertEqual(report["base_model"], "Boldt/Boldt-DC-350M")


if __name__ == "__main__":
    unittest.main()
