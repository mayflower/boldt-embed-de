"""Stdlib tests for the v5 small-RAG config loader/validator. No ML deps, no network."""
import copy
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import v5_rag_config as v5  # noqa: E402

CFG = ROOT / "configs" / "experiments" / "v5_small_rag.json"


def _cfg() -> dict:
    return json.loads(CFG.read_text("utf-8"))


def _write_tmp(obj) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(obj, f)
    f.close()
    return f.name


class TestShippedConfig(unittest.TestCase):
    def test_loads(self):
        c = v5.load_v5_rag_config(CFG)
        self.assertEqual(c.experiment_id, "v5-small-rag")
        self.assertTrue(c.legal_eval_is_diagnostic_only)
        self.assertTrue(c.public_benchmarks_eval_only)
        self.assertEqual(c.teacher_models["reranker"], "Qwen/Qwen3-Reranker-8B")
        self.assertEqual(c.teacher_models["embedding"], "Qwen/Qwen3-Embedding-8B")
        self.assertIn("faq_real", c.train_domains)
        self.assertTrue(c.dense_candidates and c.reranker_candidates)

    def test_shipped_has_no_errors(self):
        self.assertEqual(v5.validate_v5_rag(_cfg()), [])

    def test_all_success_criteria_numeric(self):
        c = v5.load_v5_rag_config(CFG)
        for k, val in c.success_criteria.items():
            self.assertTrue(isinstance(val, (int, float)) and not isinstance(val, bool), k)

    def test_local_rag_in_both_train_and_eval_is_allowed(self):
        # local_rag is private/local (not a public benchmark) -> may train and eval.
        c = v5.load_v5_rag_config(CFG)
        self.assertIn("local_rag", c.train_domains)
        self.assertIn("local_rag", c.eval_sets)
        self.assertEqual(v5.validate_v5_rag(_cfg()), [])


class TestValidationRules(unittest.TestCase):
    def test_legal_must_be_diagnostic_only(self):
        d = _cfg(); d["legal_eval_is_diagnostic_only"] = False
        self.assertTrue(any("legal_eval_is_diagnostic_only" in e for e in v5.validate_v5_rag(d)))

    def test_public_benchmarks_eval_only_must_be_true(self):
        d = _cfg(); d["public_benchmarks_eval_only"] = False
        self.assertTrue(any("public_benchmarks_eval_only" in e for e in v5.validate_v5_rag(d)))

    def test_dense_candidates_must_be_nonempty(self):
        d = _cfg(); d["dense_candidates"] = []
        self.assertTrue(any("dense_candidates" in e for e in v5.validate_v5_rag(d)))

    def test_reranker_candidates_must_be_nonempty(self):
        d = _cfg(); d["reranker_candidates"] = []
        self.assertTrue(any("reranker_candidates" in e for e in v5.validate_v5_rag(d)))

    def test_public_eval_set_cannot_appear_in_train_domains(self):
        d = _cfg(); d["train_domains"] = d["train_domains"] + ["germanquad_do_not_train"]
        errs = v5.validate_v5_rag(d)
        self.assertTrue(any("public-benchmark" in e for e in errs))

    def test_public_token_embedded_in_train_domain_is_flagged(self):
        # the WebFAQ HELD-OUT eval split must never train; WebFAQ training pairs / webfaq2 are fine
        d = _cfg(); d["train_domains"] = d["train_domains"] + ["webfaq_heldout_pairs"]
        self.assertTrue(any("public-benchmark" in e for e in v5.validate_v5_rag(d)))

    def test_webfaq_training_sources_are_not_flagged(self):
        d = _cfg(); d["train_domains"] = d["train_domains"] + ["webfaq2", "faq_real_extra"]
        self.assertEqual(v5.validate_v5_rag(d), [])   # webfaq2 / faq_real are training sources

    def test_near_ceiling_tolerance_must_be_non_positive(self):
        d = _cfg(); d["near_ceiling_eval_policy"]["use_do_not_regress_tolerance"] = 0.01
        self.assertTrue(any("use_do_not_regress_tolerance" in e for e in v5.validate_v5_rag(d)))

    def test_near_ceiling_tolerance_must_be_at_least_minus_002(self):
        d = _cfg(); d["near_ceiling_eval_policy"]["use_do_not_regress_tolerance"] = -0.05
        self.assertTrue(any("use_do_not_regress_tolerance" in e for e in v5.validate_v5_rag(d)))

    def test_near_ceiling_tolerance_zero_is_allowed(self):
        d = _cfg(); d["near_ceiling_eval_policy"]["use_do_not_regress_tolerance"] = 0
        self.assertEqual(
            [e for e in v5.validate_v5_rag(d) if "use_do_not_regress_tolerance" in e], [])

    def test_near_ceiling_must_not_be_primary_signal(self):
        d = _cfg(); d["near_ceiling_eval_policy"]["do_not_use_as_primary_promotion_signal"] = False
        self.assertTrue(
            any("do_not_use_as_primary_promotion_signal" in e for e in v5.validate_v5_rag(d)))

    def test_teacher_models_require_embedding_and_reranker(self):
        d = _cfg(); d["teacher_models"] = {"embedding": "Qwen/Qwen3-Embedding-8B"}
        self.assertTrue(any("teacher_models.reranker" in e for e in v5.validate_v5_rag(d)))

    def test_load_raises_on_invalid(self):
        d = _cfg(); d["legal_eval_is_diagnostic_only"] = False
        path = _write_tmp(d)
        with self.assertRaises(ValueError):
            v5.load_v5_rag_config(path)


if __name__ == "__main__":
    unittest.main()
