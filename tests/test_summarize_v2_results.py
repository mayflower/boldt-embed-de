"""Stdlib tests for the v2 results dashboard. No ML, no network."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
SCRIPT = ROOT / "scripts" / "summarize_v2_results.py"
V1 = FIX / "v1_results"
V2 = FIX / "v2_results"
CFG = FIX / "v2_generalization.json"


def _run(v1, v2, out, jout):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--v1-dir", str(v1), "--v2-dir", str(v2),
         "--config", str(CFG), "--output", out, "--json-output", jout],
        capture_output=True, text=True)


class TestSummary(unittest.TestCase):
    def test_improved_verdict_when_criteria_met(self):
        with tempfile.TemporaryDirectory() as d:
            md, js = f"{d}/R.md", f"{d}/R.json"
            out = _run(V1, V2, md, js)
            self.assertEqual(out.returncode, 0, out.stderr)
            res = json.loads(pathlib.Path(js).read_text())
            self.assertEqual(res["verdict"], "improved")
            self.assertEqual(res["criteria_passed"], "5/5")
            self.assertEqual(res["reranker_promotion_gate"], "pass")
            self.assertIn("# v2 results", pathlib.Path(md).read_text())

    def test_tables_and_recommendations_present(self):
        with tempfile.TemporaryDirectory() as d:
            md, js = f"{d}/R.md", f"{d}/R.json"
            _run(V1, V2, md, js)
            text = pathlib.Path(md).read_text()
            self.assertIn("Dense retrieval nDCG@10", text)
            self.assertIn("Reranker lift", text)
            self.assertIn("Matryoshka", text)
            self.assertIn("Recommendations", text)

    def test_missing_v2_warns_and_not_improved(self):
        with tempfile.TemporaryDirectory() as d:
            # empty v2-dir -> all v2 dense missing -> warnings; falls back to v1 (gerdalir 0.078 < 0.10)
            md, js = f"{d}/R.md", f"{d}/R.json"
            out = _run(V1, d, md, js)
            self.assertEqual(out.returncode, 0, out.stderr)
            res = json.loads(pathlib.Path(js).read_text())
            self.assertTrue(res["warnings"])
            self.assertNotEqual(res["verdict"], "improved")  # gerdalir below min + no reranker reports

    def test_no_ml_import(self):
        code = ("import sys; sys.path.insert(0, %r);"
                "import importlib.util as u;"
                "spec=u.spec_from_file_location('m', %r); m=u.module_from_spec(spec);"
                "assert 'torch' not in sys.modules; print('clean')") % (
                    str(ROOT / "src"), str(SCRIPT))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertIn("clean", out.stdout)


if __name__ == "__main__":
    unittest.main()
