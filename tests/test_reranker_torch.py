"""Real reranker training tests (prompt 08 'Tests'). Skipped unless torch/CUDA present."""
import pathlib
import shutil
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False

from boldt_embed import data as datamod  # noqa: E402
from boldt_embed import train  # noqa: E402
from boldt_embed.config import load_reranker_config  # noqa: E402


@unittest.skipUnless(_HAS_CUDA, "requires torch + CUDA")
class TestRealReranker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="boldt-rr-")
        cls.cfg = load_reranker_config(ROOT / "configs" / "training_reranker.json")
        cls.triples = datamod.load_jsonl(ROOT / "data" / "samples" / "toy_triples_de.jsonl")[:4]
        cls.report = train.train_reranker_real(
            cls.cfg, cls.triples, output_dir=cls.tmp, device_index=0, epochs=15,
            log=lambda *_: None)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_label_mapping(self):
        self.assertEqual(self.cfg.positive_label, "Ja")
        self.assertEqual(self.cfg.negative_label, "Nein")
        self.assertEqual(self.report["num_positive_pairs"], 4)
        self.assertGreaterEqual(self.report["num_negative_pairs"], 4)

    def test_loss_decreases(self):
        self.assertLess(self.report["final_loss"], self.report["initial_loss"])

    def test_score_monotonicity(self):
        # after overfit, score(query, positive) > score(query, hard negative)
        self.assertEqual(self.report["train_pairwise_accuracy"], 1.0)

    def test_save_load_scores(self):
        t = self.triples[0]
        scores = train.rerank_scores_real(
            self.tmp, t["query"], [t["positive"], t["negatives"][0]],
            self.cfg.input_template, device_index=0)
        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[0], scores[1])  # positive scored higher than negative


if __name__ == "__main__":
    unittest.main()
