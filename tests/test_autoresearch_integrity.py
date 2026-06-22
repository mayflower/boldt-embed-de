"""Tests for the AutoResearch protected-surface integrity checker (stdlib)."""
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("check_autoresearch_integrity")


class ClassifyTests(unittest.TestCase):
    def test_editable_surface(self):
        cls = C.classify_paths([
            "configs/autoresearch/experiments/current.json",
            "src/boldt_embed/autoresearch_recipe.py",
        ])
        self.assertEqual(len(cls["editable"]), 2)
        self.assertEqual(cls["protected"], [])

    def test_scoring_script_is_protected(self):
        self.assertIn("scripts/ar_score.py",
                      C.classify_paths(["scripts/ar_score.py"])["protected"])

    def test_eval_data_is_protected(self):
        path = "data/processed/eval/germanquad_corpus.jsonl"
        self.assertIn(path, C.classify_paths([path])["protected"])

    def test_eval_harness_and_gate_scripts_protected(self):
        cls = C.classify_paths([
            "scripts/eval_v6_1_dense_top50.py",
            "scripts/check_dense_recall_gate.py",
            "scripts/validate_release_2026.py",
        ])
        self.assertEqual(len(cls["protected"]), 3)

    def test_base_config_is_protected(self):
        # base_dense.json drives every trial via `extends`; only experiments/*.json is editable
        cls = C.classify_paths(["configs/autoresearch/base_dense.json"])
        self.assertIn("configs/autoresearch/base_dense.json", cls["protected"])

    def test_unrelated_path_is_other(self):
        cls = C.classify_paths(["docs/some-notes.md"])
        self.assertEqual(cls["other"], ["docs/some-notes.md"])
        self.assertEqual(cls["protected"], [])

    def test_leading_dot_slash_normalized(self):
        self.assertIn("scripts/ar_score.py",
                      C.classify_paths(["./scripts/ar_score.py"])["protected"])


class EvaluateTests(unittest.TestCase):
    def test_editable_only_passes(self):
        res = C.evaluate(["configs/autoresearch/experiments/current.json"])
        self.assertEqual(res["status"], "pass")
        self.assertEqual(res["violations"], [])

    def test_protected_fails(self):
        res = C.evaluate(["scripts/ar_score.py",
                          "configs/autoresearch/experiments/current.json"])
        self.assertEqual(res["status"], "fail")
        self.assertIn("scripts/ar_score.py", res["violations"])

    def test_strict_flags_other(self):
        self.assertEqual(C.evaluate(["docs/x.md"], strict=False)["status"], "pass")
        self.assertEqual(C.evaluate(["docs/x.md"], strict=True)["status"], "fail")


if __name__ == "__main__":
    unittest.main()
