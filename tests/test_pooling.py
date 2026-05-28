import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import pooling  # noqa: E402


class TestPooling(unittest.TestCase):
    hidden = [[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]]
    mask = [1, 1, 0]  # third token is padding

    def test_mean_pool_ignores_padding(self):
        self.assertEqual(pooling.mean_pool(self.hidden, self.mask), [0.5, 0.5])

    def test_last_token_pool_is_last_non_pad(self):
        self.assertEqual(pooling.last_token_pool(self.hidden, self.mask), [0.0, 1.0])

    def test_cls_pool(self):
        self.assertEqual(pooling.cls_pool(self.hidden, self.mask), [1.0, 0.0])

    def test_dispatch(self):
        self.assertEqual(pooling.pool("mean", self.hidden, self.mask), [0.5, 0.5])
        self.assertEqual(pooling.pool("eos_or_last_token", self.hidden, self.mask), [0.0, 1.0])

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            pooling.pool("banana", self.hidden, self.mask)

    def test_empty_mask_raises(self):
        with self.assertRaises(ValueError):
            pooling.mean_pool(self.hidden, [0, 0, 0])

    def test_l2_normalize(self):
        out = pooling.l2_normalize([3.0, 4.0])
        self.assertAlmostEqual(out[0], 0.6)
        self.assertAlmostEqual(out[1], 0.8)
        self.assertAlmostEqual(math.sqrt(out[0] ** 2 + out[1] ** 2), 1.0)

    def test_l2_normalize_zero_vector(self):
        self.assertEqual(pooling.l2_normalize([0.0, 0.0]), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
