"""Stdlib tests for template-based synthetic German query generation."""
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data_pipeline as dp  # noqa: E402
from boldt_embed import local_llm_generation as llm  # noqa: E402
from boldt_embed import synthetic_queries as sq  # noqa: E402

PASSAGES = ROOT / "tests" / "fixtures" / "passages.jsonl"


class TestGeneration(unittest.TestCase):
    def setUp(self):
        self.passages = list(dp.stream_jsonl(PASSAGES))

    def test_deterministic(self):
        a = sq.generate_synthetic_candidates(self.passages)
        b = sq.generate_synthetic_candidates(self.passages)
        self.assertEqual(a, b)
        self.assertTrue(a)

    def test_multiple_styles_per_passage(self):
        rows = sq.generate_queries_for_passage(self.passages[0])
        styles = {r["metadata"]["query_style"] for r in rows}
        self.assertGreaterEqual(len(styles), 4)

    def test_schema_valid(self):
        for r in sq.generate_synthetic_candidates(self.passages):
            self.assertEqual(dp.validate_candidate_record(r), [], r)
            self.assertTrue(r["positive"])
            self.assertEqual(r["source"], "synthetic")

    def test_license_and_passage_propagation(self):
        rows = sq.generate_queries_for_passage(self.passages[0])
        for r in rows:
            self.assertEqual(r["license"], "CC-BY-4.0")  # inherited from passage
            self.assertEqual(r["metadata"]["source_passage_id"], "p1")
            self.assertEqual(r["metadata"]["source_domain"], "admin")
            self.assertEqual(r["doc_id"], "p1")

    def test_queries_per_passage_cap(self):
        rows = sq.generate_queries_for_passage(self.passages[0], queries_per_passage=2)
        self.assertLessEqual(len(rows), 2)

    def test_domain_filter(self):
        rows = sq.generate_queries_for_passage(self.passages[0], domains=["legal"])
        self.assertTrue(rows)
        self.assertTrue(all(r["metadata"]["query_style"] == "legal" for r in rows))

    def test_legal_template_uses_section(self):
        rows = sq.generate_queries_for_passage(self.passages[0], domains=["legal"])
        self.assertTrue(any("§ 551" in r["query"] for r in rows))


class TestLocalLLMStub(unittest.TestCase):
    def test_instance_method_raises(self):
        gen = llm.LocalLLMGenerator(model_name="x")
        with self.assertRaises(NotImplementedError):
            gen.generate_queries_with_local_model({"document": "x"})

    def test_module_fn_raises(self):
        with self.assertRaises(NotImplementedError):
            llm.generate_queries_with_local_model({"document": "x"})


class TestNoExternalCalls(unittest.TestCase):
    def test_generation_imports_no_network_or_ml(self):
        code = (
            "import sys; sys.path.insert(0, %r);"
            "from boldt_embed import synthetic_queries as sq, data_pipeline as dp;"
            "rows = sq.generate_synthetic_candidates(list(dp.stream_jsonl(%r)));"
            "assert rows;"
            "assert 'torch' not in sys.modules;"
            "assert 'requests' not in sys.modules;"
            "assert 'urllib.request' not in sys.modules;"
            "print('OK')"
        ) % (str(ROOT / "src"), str(PASSAGES))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("OK", out.stdout)

    def test_script_dry_run(self):
        out = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_synthetic_queries.py"),
             "--passages", str(PASSAGES), "--dry-run"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("DRY RUN", out.stdout)


if __name__ == "__main__":
    unittest.main()
