"""Real bidirectional/MNTP tests (prompt 07 'Tests'). Skipped unless torch/CUDA present."""
import pathlib
import shutil
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_TORCH = False
    _HAS_CUDA = False

from boldt_embed import data as datamod  # noqa: E402
from boldt_embed import train  # noqa: E402
from boldt_embed.config import load_bidirectional_config  # noqa: E402


@unittest.skipUnless(_HAS_TORCH, "requires torch")
class TestMNTPMasking(unittest.TestCase):
    def test_mask_tokens_labels_and_replacement(self):
        ids = torch.arange(2, 22).reshape(2, 10)  # no special ids in range
        attn = torch.ones(2, 10, dtype=torch.long)
        masked_input, labels, masked = train.mask_tokens(ids, attn, 0.5, vocab_size=32000,
                                                          special_ids=[0, 1])
        # labels are -100 exactly where NOT masked
        self.assertTrue(((labels == -100) == (~masked)).all().item())
        # at masked positions labels hold the ORIGINAL id
        self.assertTrue((labels[masked] == ids[masked]).all().item())
        # some positions were masked given prob 0.5 over 20 tokens
        self.assertGreater(int(masked.sum()), 0)


@unittest.skipUnless(_HAS_CUDA, "requires CUDA")
class TestBidirectionalReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="boldt-bi-")
        cfg = load_bidirectional_config(ROOT / "configs" / "training_bidirectional.json")
        triples = datamod.load_jsonl(ROOT / "data" / "samples" / "toy_triples_de.jsonl")[:3]
        texts = [t["positive"] for t in triples]
        cls.report = train.train_bidirectional_real(
            cfg, triples, texts, output_dir=cls.tmp, device_index=0,
            mntp_steps=3, contrastive_steps=5, log=lambda *_: None)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_attention_is_bidirectional(self):
        # token-0 hidden state must change when a LATER token changes (impossible if causal)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.tmp, trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(
            self.tmp, trust_remote_code=True, torch_dtype=torch.float32,
            attn_implementation="eager").to("cuda:0").eval()
        train.enable_bidirectional(m)
        a = tok("der hund laeuft schnell", return_tensors="pt").input_ids.to("cuda:0")
        b = a.clone()
        b[0, -1] = (b[0, -1] + 5)
        with torch.no_grad():
            ha = m.model(input_ids=a).last_hidden_state[0, 0]
            hb = m.model(input_ids=b).last_hidden_state[0, 0]
        self.assertGreater(float((ha - hb).abs().max()), 1e-3)

    def test_mntp_and_contrastive_ran(self):
        self.assertEqual(len(self.report["mntp_loss_curve"]), 3)
        self.assertEqual(len(self.report["contrastive_loss_curve"]), 5)
        self.assertEqual(self.report["hidden_size"], 1024)

    def test_contrastive_loss_decreases(self):
        self.assertLessEqual(self.report["contrastive_final_loss"],
                             self.report["contrastive_initial_loss"] + 1e-6)

    def test_save_load(self):
        vecs = train.encode_texts(self.tmp, ["Ein Satz."], pooling="mean", device_index=0)
        self.assertEqual(len(vecs[0]), 1024)


if __name__ == "__main__":
    unittest.main()
