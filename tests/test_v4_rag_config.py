"""Stdlib tests for the v4 RAG-reranker config loader/validator. No ML deps, no network."""
import copy
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import v4_rag_config as v4  # noqa: E402

CFG = ROOT / "configs" / "experiments" / "v4_rag_reranker.json"


def _write_tmp(obj) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(obj, f); f.close()
    return f.name


class TestShippedConfig(unittest.TestCase):
    def test_loads(self):
        c = v4.load_v4_rag_config(CFG)
        self.assertEqual(c.experiment_id, "v4-rag-reranker")
        self.assertTrue(c.legal_eval_is_diagnostic_only)
        self.assertTrue(c.public_benchmarks_eval_only)
        self.assertEqual(c.teacher_reranker, "Qwen/Qwen3-Reranker-8B")
        self.assertIn("bm25", c.candidate_sources)
        self.assertIn("faq_real", c.train_domains)

    def test_shipped_has_no_errors(self):
        self.assertEqual(v4.validate_v4_rag(json.loads(CFG.read_text("utf-8"))), [])

    def test_all_success_criteria_numeric(self):
        c = v4.load_v4_rag_config(CFG)
        for k, val in c.success_criteria.items():
            self.assertTrue(isinstance(val, (int, float)) and not isinstance(val, bool), k)

    def test_train_domains_have_no_eval_sources(self):
        c = v4.load_v4_rag_config(CFG)
        self.assertEqual(set(c.train_domains) & v4.PUBLIC_BENCHMARK_EVAL, set())

    def test_loader_is_stdlib_only(self):
        code = ("import sys; sys.path.insert(0, %r);"
                "from boldt_embed import v4_rag_config as v4;"
                "v4.load_v4_rag_config(%r);"
                "assert 'torch' not in sys.modules; print('clean')") % (str(ROOT / "src"), str(CFG))
        out = subprocess_run(code)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.base = json.loads(CFG.read_text("utf-8"))

    def test_legal_diagnostic_must_be_true(self):
        d = copy.deepcopy(self.base); d["legal_eval_is_diagnostic_only"] = False
        self.assertTrue(any("legal_eval_is_diagnostic_only" in e for e in v4.validate_v4_rag(d)))

    def test_public_eval_only_must_be_true(self):
        d = copy.deepcopy(self.base); d["public_benchmarks_eval_only"] = False
        self.assertTrue(any("public_benchmarks_eval_only" in e for e in v4.validate_v4_rag(d)))

    def test_train_domains_with_eval_source_fails(self):
        d = copy.deepcopy(self.base); d["train_domains"] = ["faq_real", "germanquad"]
        errs = v4.validate_v4_rag(d)
        self.assertTrue(any("public-benchmark/eval" in e for e in errs), errs)

    def test_empty_candidate_sources_fails(self):
        d = copy.deepcopy(self.base); d["candidate_sources"] = []
        self.assertTrue(any("candidate_sources" in e for e in v4.validate_v4_rag(d)))

    def test_nonnumeric_success_criterion_fails(self):
        d = copy.deepcopy(self.base)
        d["success_criteria"]["webfaq_reranker_delta_ndcg10_min"] = "high"
        self.assertTrue(any("success_criteria" in e for e in v4.validate_v4_rag(d)))

    def test_load_raises_on_invalid(self):
        d = copy.deepcopy(self.base); d["legal_eval_is_diagnostic_only"] = False
        with self.assertRaises(ValueError):
            v4.load_v4_rag_config(_write_tmp(d))


def subprocess_run(code):
    import subprocess
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
