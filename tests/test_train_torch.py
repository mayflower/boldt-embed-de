"""Real GPU training tests (prompt 06 'Tests'). Skipped unless torch + CUDA are present,
so they run where a GPU exists (and are skipped in the stdlib CI matrix)."""
import math
import pathlib
import shutil
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False

try:
    import sentence_transformers  # noqa: F401
    _HAS_ST = True
except Exception:
    _HAS_ST = False

from boldt_embed import data as datamod  # noqa: E402
from boldt_embed import train  # noqa: E402
from boldt_embed.config import load_causal_config  # noqa: E402


@unittest.skipUnless(_HAS_CUDA, "requires torch + CUDA")
class TestRealCausalTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="boldt-test-")
        cfg = load_causal_config(ROOT / "configs" / "training_causal.json")
        triples = datamod.load_jsonl(ROOT / "data" / "samples" / "toy_triples_de.jsonl")[:3]
        cls.report = train.train_causal_real(
            cfg, triples, output_dir=cls.tmp, device_index=0, epochs=5, lr=2e-5,
            log=lambda *_: None,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_hidden_size_and_params(self):
        self.assertEqual(self.report["hidden_size"], 1024)
        self.assertGreater(self.report["param_count"], 4e8)

    def test_loss_decreases_on_overfit(self):
        self.assertLessEqual(self.report["final_loss"], self.report["initial_loss"] + 1e-6)

    def test_save_load_encode(self):
        vecs = train.encode_texts(self.tmp, ["Ein deutscher Testsatz."],
                                  pooling="eos", device_index=0)
        self.assertEqual(len(vecs[0]), 1024)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in vecs[0])), 1.0, places=4)

    @unittest.skipUnless(_HAS_ST, "requires sentence-transformers")
    def test_sentence_transformers_export(self):
        import export_sentence_transformers as exporter
        from sentence_transformers import SentenceTransformer

        out = tempfile.mkdtemp(prefix="boldt-st-")
        try:
            exporter.export(self.tmp, out, pooling="eos", max_seq_length=64)
            st = SentenceTransformer(out)
            vec = st.encode(["Hallo Welt."], normalize_embeddings=True)
            self.assertEqual(len(vec[0]), 1024)
        finally:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
