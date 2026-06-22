"""Tests for the stdlib EmbedFilter core (no ML; spec selection + metadata validation)."""
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import embed_filter as EF  # noqa: E402


class SelectBulkSliceTests(unittest.TestCase):
    def test_keep_dims_and_centered_slices(self):
        expect = {
            1:  (1024, 0, 1024),
            2:  (512, 256, 768),
            4:  (256, 384, 640),
            8:  (128, 448, 576),
            16: (64, 480, 544),
        }
        for tau, (keep, left, right) in expect.items():
            spec = EF.select_bulk_slice(1024, tau)
            self.assertEqual((spec.keep_dim, spec.left, spec.right), (keep, left, right),
                             f"tau={tau}")
            self.assertEqual(spec.right - spec.left, spec.keep_dim)
            self.assertEqual(spec.hidden_dim, 1024)
            self.assertEqual(spec.strategy, "bulk_center")

    def test_tau1_keeps_all(self):
        spec = EF.select_bulk_slice(768, 1)
        self.assertEqual((spec.keep_dim, spec.left, spec.right), (768, 0, 768))

    def test_invalid_tau_raises(self):
        for bad in (3, 5, 0, 32, -2):
            with self.assertRaises(ValueError):
                EF.select_bulk_slice(1024, bad)

    def test_non_divisible_raises(self):
        with self.assertRaises(ValueError):
            EF.select_bulk_slice(1000, 16)   # 1000 % 16 != 0

    def test_bad_hidden_dim_raises(self):
        for bad in (0, -1, 3.5, True):
            with self.assertRaises(ValueError):
                EF.select_bulk_slice(bad, 2)


class MetadataTests(unittest.TestCase):
    def _good(self):
        spec = EF.select_bulk_slice(1024, 4)
        return EF.metadata_for_spec(spec, model="Boldt/Boldt-DC-350M",
                                    source_matrix="lm_head", vocab_size=32000)

    def test_valid_metadata_passes(self):
        self.assertEqual(EF.validate_embed_filter_metadata(self._good()), [])

    def test_wrong_hidden_dim_caught(self):
        m = self._good(); m["hidden_dim"] = 0
        self.assertTrue(any("hidden_dim" in e for e in EF.validate_embed_filter_metadata(m)))

    def test_bad_bounds_caught(self):
        m = self._good(); m["right"] = m["hidden_dim"] + 50   # right > H
        probs = EF.validate_embed_filter_metadata(m)
        self.assertTrue(any("bounds" in e or "keep_dim" in e for e in probs))

    def test_keep_dim_mismatch_caught(self):
        m = self._good(); m["keep_dim"] = 999
        self.assertTrue(EF.validate_embed_filter_metadata(m))

    def test_bad_source_matrix_caught(self):
        m = self._good(); m["source_matrix"] = "something_else"
        self.assertTrue(any("source_matrix" in e for e in EF.validate_embed_filter_metadata(m)))

    def test_non_dict_metadata(self):
        self.assertTrue(EF.validate_embed_filter_metadata(["not", "a", "dict"]))


class EncoderConflictTests(unittest.TestCase):
    """`dim` + `embed_filter` must raise BEFORE any torch import (competing reductions)."""

    def test_causal_conflict_raises_without_torch(self):
        from boldt_embed.model_causal import CausalEmbedder
        emb = CausalEmbedder.from_config(str(ROOT / "configs" / "training_causal.json"))
        with self.assertRaises(ValueError):
            emb.encode(["frage"], dim=256, embed_filter="outputs/embedfilter/x")

    def test_bidirectional_conflict_raises_without_torch(self):
        from boldt_embed.model_bidirectional import BidirectionalEmbedder
        cfg = str(ROOT / "configs" / "training_bidirectional.json")
        emb = BidirectionalEmbedder.from_config(cfg)
        with self.assertRaises(ValueError):
            emb.encode(["frage"], dim=256, embed_filter="outputs/embedfilter/x")


class LoadBasisTorchTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch not available")

    def test_round_trip_and_mismatch(self):
        import json
        import tempfile

        import torch
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            torch.save(torch.randn(8, 4), d / "basis.pt")
            spec = EF.select_bulk_slice(8, 2)               # keep_dim 4
            (d / "metadata.json").write_text(json.dumps(
                EF.metadata_for_spec(spec, model="m", source_matrix="lm_head", vocab_size=10)),
                encoding="utf-8")
            basis, meta = EF.load_embed_filter_basis(str(d), expected_hidden_dim=8)
            self.assertEqual(tuple(basis.shape), (8, 4))
            self.assertEqual(meta["keep_dim"], 4)
            with self.assertRaises(ValueError):             # hidden-dim mismatch
                EF.load_embed_filter_basis(str(d), expected_hidden_dim=16)


class ImportSafetyTests(unittest.TestCase):
    def test_import_does_not_pull_in_torch(self):
        # order-independent: a fresh interpreter importing only embed_filter must not load torch
        code = ("import sys; sys.path.insert(0, 'src'); "
                "import boldt_embed.embed_filter; "
                "assert 'torch' not in sys.modules, 'torch imported at module load'")
        r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
