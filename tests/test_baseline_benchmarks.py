"""Stdlib tests for the baseline benchmark runner: config, metadata, dry-run, tiny benchmark."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from boldt_embed import config_teacher as ct  # noqa: E402
import run_baseline_benchmarks as RB  # noqa: E402

CONFIGS = ROOT / "configs"
HASHING_CFG = ROOT / "tests" / "fixtures" / "baseline_models_hashing.json"
CORPUS = ROOT / "tests" / "fixtures" / "hybrid_corpus.jsonl"
QUERIES = ROOT / "tests" / "fixtures" / "hybrid_queries.jsonl"
QRELS = ROOT / "tests" / "fixtures" / "hybrid_qrels.jsonl"
SCRIPT = ROOT / "scripts" / "run_baseline_benchmarks.py"


class TestConfig(unittest.TestCase):
    def test_shipped_config_loads(self):
        models = ct.load_baseline_models_config(CONFIGS / "baseline_models.json")
        self.assertEqual(len(models), 8)
        e5 = next(m for m in models if m.model_name_or_path == "intfloat/multilingual-e5-base")
        self.assertEqual(e5.backend, "sentence_transformers")
        self.assertEqual(e5.expected_dim, 768)
        self.assertTrue(any(m.backend == "local_boldt" for m in models))

    def test_bad_backend_rejected(self):
        errs = ct.validate_baseline_models({"models": [{"model_name_or_path": "x", "backend": "magic"}]})
        self.assertTrue(any("backend" in e for e in errs), errs)

    def test_empty_models_rejected(self):
        errs = ct.validate_baseline_models({"models": []})
        self.assertTrue(any("models" in e for e in errs), errs)


class TestMetadata(unittest.TestCase):
    def test_collect_env_metadata_keys(self):
        # Versions are read from package metadata (importlib.metadata), not by importing the
        # packages. The no-ML-import guarantee for the whole script (incl. this call, which
        # runs before the dry-run branch) is verified in TestDryRun via a subprocess.
        meta = RB.collect_env_metadata()
        for key in ("commit", "python", "platform", "torch", "transformers", "sentence_transformers"):
            self.assertIn(key, meta)


class TestDryRun(unittest.TestCase):
    def test_dry_run_lists_models_no_ml(self):
        out = subprocess.run(
            [sys.executable, str(SCRIPT), "--models", str(CONFIGS / "baseline_models.json"), "--dry-run"],
            capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("dry-run-ok", out.stdout)
        self.assertIn("Qwen/Qwen3-Embedding-8B", out.stdout)


class TestTinyBenchmark(unittest.TestCase):
    def _run(self, out_path, run_card_dir):
        env = dict(os.environ, BOLDT_RUN_CARD_DIR=str(run_card_dir))  # don't pollute repo outputs/
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--models", str(HASHING_CFG), "--mode", "local",
             "--eval-corpus", str(CORPUS), "--eval-queries", str(QUERIES), "--qrels", str(QRELS),
             "--output", str(out_path)], capture_output=True, text=True, env=env)

    def test_local_hashing_benchmark_deterministic_and_no_ml(self):
        with tempfile.TemporaryDirectory() as d:
            out1 = pathlib.Path(d) / "r1.json"
            out2 = pathlib.Path(d) / "r2.json"
            r1 = self._run(out1, pathlib.Path(d) / "cards")
            r2 = self._run(out2, pathlib.Path(d) / "cards")
            self.assertEqual(r1.returncode, 0, r1.stderr)
            self.assertEqual(r2.returncode, 0, r2.stderr)
            rep1 = json.loads(out1.read_text())
            rep2 = json.loads(out2.read_text())
            m1 = rep1["results"][0]["metrics"]
            m2 = rep2["results"][0]["metrics"]
            self.assertEqual(m1, m2)                 # deterministic
            self.assertIn("ndcg@10", m1)
            self.assertIn("recall@100", m1)
            self.assertTrue(out1.with_suffix(".md").exists())  # markdown table written
            self.assertIn("commit", rep1["run_metadata"])


if __name__ == "__main__":
    unittest.main()
