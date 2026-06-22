"""Tests for the unembedding-lens diagnostic (token classification + dry-run; no ML)."""
import contextlib
import importlib.util
import io
import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load("diagnose_unembedding_lens")


class TokenCategoryTests(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(D.token_category(","), "punctuation")
        self.assertEqual(D.token_category("."), "punctuation")
        self.assertEqual(D.token_category("▁der"), "stopword")
        self.assertEqual(D.token_category("▁und"), "stopword")
        self.assertEqual(D.token_category("ung"), "subword")       # continuation fragment
        self.assertEqual(D.token_category("▁Datenschutz"), "content")

    def test_noncontent_ratio(self):
        toks = ["▁der", ",", "ung", "▁Haus"]   # 3 non-content, 1 content
        self.assertEqual(D.noncontent_ratio(toks), 0.75)
        self.assertEqual(D.noncontent_ratio([]), 0.0)


class DryRunTests(unittest.TestCase):
    def test_dry_run_ok(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = D.main(["--dry-run", "--model", "Boldt/Boldt-DC-350M", "--top-k", "20"])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(buf.getvalue())["status"], "dry_run")

    def test_dry_run_imports_no_ml(self):
        code = (
            "import sys, importlib.util;"
            "spec=importlib.util.spec_from_file_location('d',"
            "'scripts/diagnose_unembedding_lens.py');"
            "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
            "rc=m.main(['--dry-run','--model','x','--top-k','5']); assert rc==0, rc;"
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
        )
        r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
