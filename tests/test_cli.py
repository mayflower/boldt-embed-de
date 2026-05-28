import io
import pathlib
import sys
import unittest
from contextlib import redirect_stdout

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import cli  # noqa: E402


class TestCli(unittest.TestCase):
    def test_validate(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main_validate()
        self.assertEqual(rc, 0)
        self.assertIn("pass", buf.getvalue())

    def test_bench(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main_bench()
        self.assertEqual(rc, 0)

    def test_smoke(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main_smoke()
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
