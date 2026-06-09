"""Stdlib tests for the bidirectional adapter's pooling/delta math + dry-run (no ML)."""
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import llm2vec_boldt as B  # noqa: E402

MNTP = ROOT / "tests" / "fixtures" / "mntp_texts.jsonl"


class TestPoolingMath(unittest.TestCase):
    def test_masked_mean_pool(self):
        hidden = [[1.0, 1.0], [3.0, 3.0], [9.0, 9.0]]
        mask = [1, 1, 0]  # last token padded -> ignored
        self.assertEqual(B.masked_mean_pool(hidden, mask), [2.0, 2.0])

    def test_last_token_pool_uses_last_unmasked(self):
        hidden = [[1.0, 1.0], [3.0, 3.0], [9.0, 9.0]]
        mask = [1, 1, 0]
        self.assertEqual(B.last_token_pool(hidden, mask), [3.0, 3.0])

    def test_pool_embeddings_list_path_shapes(self):
        hidden = [[[1.0, 1.0], [3.0, 3.0]], [[2.0, 4.0], [4.0, 8.0]]]  # [B=2,T=2,H=2]
        mask = [[1, 1], [1, 1]]
        pooled = B.pool_embeddings(hidden, mask, pooling="mean")
        self.assertEqual(len(pooled), 2)            # batch
        self.assertEqual(len(pooled[0]), 2)         # hidden
        self.assertEqual(pooled[0], [2.0, 2.0])
        self.assertEqual(pooled[1], [3.0, 6.0])

    def test_pool_embeddings_eos_list_path(self):
        hidden = [[[1.0], [5.0], [9.0]]]
        mask = [[1, 1, 0]]
        self.assertEqual(B.pool_embeddings(hidden, mask, pooling="eos"), [[5.0]])

    def test_pooled_output_shape(self):
        self.assertEqual(B.pooled_output_shape(8, 1024), (8, 1024))


class TestDeltaMath(unittest.TestCase):
    def test_l2_delta(self):
        self.assertEqual(B.l2_delta([0.0, 0.0], [3.0, 4.0]), 5.0)
        self.assertEqual(B.l2_delta([1.0, 2.0], [1.0, 2.0]), 0.0)


class TestNoMLImport(unittest.TestCase):
    def test_module_and_pooling_import_no_torch(self):
        code = (
            "import sys; sys.path.insert(0, %r);"
            "from boldt_embed import llm2vec_boldt as B;"
            "B.pool_embeddings([[[1.0,1.0],[3.0,3.0]]], [[1,1]], 'mean');"
            "assert 'torch' not in sys.modules;"
            "print('clean')"
        ) % str(ROOT / "src")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


class TestDryRunCLI(unittest.TestCase):
    def test_dry_run(self):
        out = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "prepare_bidirectional_student.py"),
             "--texts", str(MNTP), "--steps", "10", "--dry-run"],
            capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("dry-run-ok", out.stdout)
        self.assertIn("num_mntp_texts", out.stdout)


if __name__ == "__main__":
    unittest.main()
