"""The train module must import without torch (torch is used lazily inside functions)."""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import train  # noqa: E402


class TestTrainImportable(unittest.TestCase):
    def test_callables_present(self):
        for name in ("pick_device", "info_nce", "train_causal_real", "encode_texts"):
            self.assertTrue(callable(getattr(train, name)), name)


if __name__ == "__main__":
    unittest.main()
