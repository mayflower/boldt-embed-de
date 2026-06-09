"""Stdlib tests for the teacher/student-2026 config layer. No ML deps, no network."""
import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import config_teacher as ct  # noqa: E402

CONFIGS = ROOT / "configs"


def _write_tmp(obj) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(obj, f)
    f.close()
    return f.name


class TestShippedConfigs(unittest.TestCase):
    def test_teacher_models_loads(self):
        cfg = ct.load_teacher_models_config(CONFIGS / "teacher_models.json")
        self.assertEqual(cfg.embedding_teacher.model_name, "Qwen/Qwen3-Embedding-8B")
        self.assertEqual(cfg.embedding_teacher.backend, "sentence_transformers")
        self.assertEqual(cfg.embedding_teacher.output_dim, 1024)
        self.assertTrue(cfg.embedding_teacher.normalize)
        self.assertEqual(cfg.reranker_teacher.model_name, "Qwen/Qwen3-Reranker-8B")
        self.assertEqual(cfg.reranker_teacher.backend, "sentence_transformers_cross_encoder")
        self.assertEqual(cfg.reranker_teacher.score_activation, "raw")

    def test_student_training_loads(self):
        cfg = ct.load_student_training_config(CONFIGS / "student_training_2026.json")
        self.assertEqual(cfg.base_model, "Boldt/Boldt-DC-350M")
        self.assertEqual(cfg.student_variant, "bidirectional")
        self.assertEqual(cfg.matryoshka_dims[0], 1024)
        self.assertEqual(cfg.target_dim, 1024)
        self.assertIn("matryoshka", cfg.losses)
        self.assertEqual(cfg.train_eval_split_policy, "public_benchmarks_eval_only")

    def test_loaders_do_not_import_ml(self):
        # Run in a clean subprocess: checking this process's sys.modules is unreliable
        # because sibling test modules import torch first. The contract is that loading
        # configs must never pull in ML libraries.
        code = (
            "import sys; sys.path.insert(0, %r);"
            "from boldt_embed import config_teacher as ct;"
            "ct.load_teacher_models_config(%r);"
            "ct.load_student_training_config(%r);"
            "assert 'torch' not in sys.modules, 'torch imported';"
            "assert 'sentence_transformers' not in sys.modules, 'sentence_transformers imported';"
            "print('clean')"
        ) % (
            str(ROOT / "src"),
            str(CONFIGS / "teacher_models.json"),
            str(CONFIGS / "student_training_2026.json"),
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


class TestTeacherValidation(unittest.TestCase):
    def setUp(self):
        self.base = json.loads((CONFIGS / "teacher_models.json").read_text(encoding="utf-8"))

    def test_missing_embedding_section_errors(self):
        d = copy.deepcopy(self.base)
        del d["embedding_teacher"]
        errs = ct.validate_teacher_models(d)
        self.assertTrue(any("embedding_teacher" in e for e in errs), errs)

    def test_missing_required_field_errors_usefully(self):
        d = copy.deepcopy(self.base)
        del d["embedding_teacher"]["model_name"]
        errs = ct.validate_teacher_models(d)
        self.assertTrue(any("model_name" in e for e in errs), errs)

    def test_unknown_backend_rejected(self):
        d = copy.deepcopy(self.base)
        d["reranker_teacher"]["backend"] = "magic"
        errs = ct.validate_teacher_models(d)
        self.assertTrue(any("backend" in e for e in errs), errs)

    def test_bad_dtype_rejected(self):
        d = copy.deepcopy(self.base)
        d["embedding_teacher"]["torch_dtype"] = "int4"
        errs = ct.validate_teacher_models(d)
        self.assertTrue(any("torch_dtype" in e for e in errs), errs)

    def test_load_raises_on_invalid(self):
        d = copy.deepcopy(self.base)
        d["reranker_teacher"]["max_length"] = -1
        path = _write_tmp(d)
        with self.assertRaises(ValueError):
            ct.load_teacher_models_config(path)


class TestStudentValidation(unittest.TestCase):
    def setUp(self):
        self.base = json.loads((CONFIGS / "student_training_2026.json").read_text(encoding="utf-8"))

    def test_ascending_matryoshka_rejected(self):
        d = copy.deepcopy(self.base)
        d["matryoshka_dims"] = [64, 128, 256]
        errs = ct.validate_student_training(d)
        self.assertTrue(any("strictly decreasing" in e for e in errs), errs)

    def test_target_dim_must_be_in_dims(self):
        d = copy.deepcopy(self.base)
        d["target_dim"] = 999
        errs = ct.validate_student_training(d)
        self.assertTrue(any("target_dim" in e for e in errs), errs)

    def test_unknown_variant_rejected(self):
        d = copy.deepcopy(self.base)
        d["student_variant"] = "diffusion"
        errs = ct.validate_student_training(d)
        self.assertTrue(any("student_variant" in e for e in errs), errs)

    def test_leakage_policy_enforced(self):
        d = copy.deepcopy(self.base)
        d["train_eval_split_policy"] = "train_on_everything"
        errs = ct.validate_student_training(d)
        self.assertTrue(any("eval-only" in e for e in errs), errs)

    def test_empty_losses_rejected(self):
        d = copy.deepcopy(self.base)
        d["losses"] = []
        errs = ct.validate_student_training(d)
        self.assertTrue(any("losses" in e for e in errs), errs)


if __name__ == "__main__":
    unittest.main()
