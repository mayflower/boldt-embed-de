import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import merging  # noqa: E402


class TestMerging(unittest.TestCase):
    def test_lerp_endpoints_and_midpoint(self):
        a, b = [0.0, 0.0], [2.0, 4.0]
        self.assertEqual(merging.lerp(a, b, 0.0), a)
        self.assertEqual(merging.lerp(a, b, 1.0), b)
        self.assertEqual(merging.lerp(a, b, 0.5), [1.0, 2.0])

    def test_slerp_endpoints(self):
        a, b = [1.0, 0.0], [0.0, 1.0]
        self.assertAlmostEqual(merging.slerp(a, b, 0.0)[0], 1.0)
        self.assertAlmostEqual(merging.slerp(a, b, 1.0)[1], 1.0)

    def test_slerp_preserves_unit_norm(self):
        a, b = [1.0, 0.0], [0.0, 1.0]
        mid = merging.slerp(a, b, 0.5)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in mid)), 1.0)
        # lerp midpoint of orthogonal unit vectors has norm < 1
        lerp_mid = merging.lerp(a, b, 0.5)
        self.assertLess(math.sqrt(sum(x * x for x in lerp_mid)), 1.0)

    def test_slerp_colinear_falls_back_to_lerp(self):
        a, b = [1.0, 0.0], [2.0, 0.0]
        self.assertEqual(merging.slerp(a, b, 0.5), merging.lerp(a, b, 0.5))

    def test_merge_dispatch_and_unknown(self):
        self.assertEqual(merging.merge("linear", [0.0], [2.0], 0.5), [1.0])
        with self.assertRaises(ValueError):
            merging.merge("bogus", [0.0], [1.0])


if __name__ == "__main__":
    unittest.main()
