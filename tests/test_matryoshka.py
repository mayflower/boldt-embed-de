import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import matryoshka  # noqa: E402


class TestMatryoshka(unittest.TestCase):
    vec = [3.0, 4.0, 0.0, 0.0]

    def test_truncate(self):
        self.assertEqual(matryoshka.truncate(self.vec, 2), [3.0, 4.0])

    def test_truncate_normalized_unit_norm(self):
        out = matryoshka.truncate_normalized(self.vec, 2)
        self.assertAlmostEqual(out[0], 0.6)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in out)), 1.0)

    def test_truncate_too_large_raises(self):
        with self.assertRaises(ValueError):
            matryoshka.truncate(self.vec, 99)

    def test_views_each_unit_norm(self):
        views = matryoshka.matryoshka_views([1.0, 1.0, 1.0, 1.0], [4, 2])
        self.assertEqual(set(views), {4, 2})
        for v in views.values():
            self.assertAlmostEqual(math.sqrt(sum(x * x for x in v)), 1.0)


if __name__ == "__main__":
    unittest.main()
