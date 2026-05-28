import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import losses  # noqa: E402


class TestCosine(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(losses.cosine_similarity([1.0, 1.0], [2.0, 2.0]), 1.0)

    def test_orthogonal(self):
        self.assertAlmostEqual(losses.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_zero_vector(self):
        self.assertEqual(losses.cosine_similarity([0.0, 0.0], [1.0, 0.0]), 0.0)


class TestInfoNCE(unittest.TestCase):
    q = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def test_aligned_batch_low_loss(self):
        loss = losses.info_nce_loss(self.q, self.q)
        self.assertLess(loss, 1e-3)

    def test_misaligned_batch_higher_loss(self):
        wrong = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        good = losses.info_nce_loss(self.q, self.q)
        bad = losses.info_nce_loss(self.q, wrong)
        self.assertGreater(bad, good + 1.0)

    def test_close_hard_negative_raises_loss(self):
        q = [[1.0, 0.0, 0.0]]
        pos = [[1.0, 0.0, 0.0]]
        near = [[0.8, 0.6, 0.0]]   # cosine 0.8 with the query
        far = [[0.0, 1.0, 0.0]]    # cosine 0.0 with the query
        loss_near = losses.info_nce_loss(q, pos, hard_negatives=[near])
        loss_far = losses.info_nce_loss(q, pos, hard_negatives=[far])
        self.assertGreater(loss_near, loss_far)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            losses.info_nce_loss(self.q, self.q[:2])

    def test_bad_temperature_raises(self):
        with self.assertRaises(ValueError):
            losses.info_nce_loss(self.q, self.q, temperature=0.0)


if __name__ == "__main__":
    unittest.main()
