"""Tests for the dense AutoResearch recipe (dry-run is pure stdlib; real-mode safe-adapter)."""
import importlib.util
import pathlib
import sys
import tempfile
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import autoresearch_recipe as R  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load_script("ar_score")


def _config(seed=1337, pooling="mean"):
    return {
        "task": "dense_retriever",
        "seed": seed,
        "pooling": pooling,
        "normalize_embeddings": True,
        "matryoshka_dims": [1024, 512, 256, 128],
        "data_mixture": {"mmarco_de": 0.5, "webfaq_train": 0.5},
        "loss": {"type": "cached_mnrl_matryoshka_distillation", "temperature": 0.03},
        "training": {"batch_size": 32, "learning_rate": 2e-5},
        "runtime": {"dry_run": True},
    }


class DryRunTests(unittest.TestCase):
    def test_deterministic_same_seed(self):
        with tempfile.TemporaryDirectory() as d:
            a = R.run_dense_trial(_config(), pathlib.Path(d) / "a", 0.0, dry_run=True)
            b = R.run_dense_trial(_config(), pathlib.Path(d) / "b", 0.0, dry_run=True)
            self.assertEqual(a["metrics"], b["metrics"])
            self.assertEqual(a["status"], "ok")
            self.assertIn("plumbing only", a["scale_disclaimer"])

    def test_changes_when_config_changes(self):
        with tempfile.TemporaryDirectory() as d:
            a = R.run_dense_trial(_config(seed=1), pathlib.Path(d) / "a", 0.0, dry_run=True)
            b = R.run_dense_trial(_config(seed=2), pathlib.Path(d) / "b", 0.0, dry_run=True)
            self.assertNotEqual(a["metrics"], b["metrics"])
            c = R.run_dense_trial(_config(pooling="eos_or_last_token"),
                                  pathlib.Path(d) / "c", 0.0, dry_run=True)
            self.assertNotEqual(a["metrics"], c["metrics"])

    def test_writes_recipe_plan(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "run"
            R.run_dense_trial(_config(), out, 0.0, dry_run=True)
            self.assertTrue((out / "recipe_plan.json").exists())

    def test_metrics_schema_compatible_with_scorer(self):
        with tempfile.TemporaryDirectory() as d:
            res = R.run_dense_trial(_config(), pathlib.Path(d) / "run", 0.0, dry_run=True)
            scored = S.score_run(res, res)   # self-compare: all deltas 0
            for key in ("webfaq_recall@100", "webfaq_ndcg@10", "germanquad_ndcg@10",
                        "dt_test_ndcg@10"):
                self.assertEqual(scored["deltas"][key], 0.0)
            self.assertIn("failed_gates", scored)

    def test_invalid_config_returns_fail(self):
        cfg = _config()
        cfg["data_mixture"] = {"a": 0.4, "b": 0.4}   # sums to 0.8, not 1.0
        with tempfile.TemporaryDirectory() as d:
            res = R.run_dense_trial(cfg, pathlib.Path(d) / "run", 0.0, dry_run=True)
            self.assertEqual(res["status"], "fail")
            self.assertIn("data_mixture", res["note"])


class PlanAndHelperTests(unittest.TestCase):
    def test_max_seq_length_is_larger_of_query_and_doc(self):
        plan = R.build_training_plan({"training": {"max_query_length": 256,
                                                   "max_document_length": 1024}})
        self.assertEqual(plan["max_seq_length"], 1024)   # documents not truncated to query len

    def test_looks_local_vs_hf_id(self):
        self.assertTrue(R._looks_local("outputs/v6-dense-rag/checkpoints/x"))
        self.assertTrue(R._looks_local("./checkpoints/x"))
        self.assertFalse(R._looks_local("Boldt/Boldt-DC-350M"))   # remote HF id

    def test_resolve_model_path(self):
        self.assertTrue(R._resolve_model_path("outputs/x").endswith("/outputs/x")
                        or R._resolve_model_path("outputs/x").endswith("\\outputs\\x"))
        self.assertEqual(R._resolve_model_path("intfloat/multilingual-e5-base"),
                         "intfloat/multilingual-e5-base")   # HF id unchanged

    def test_leakage_from_manifest_absent_is_not_checked(self):
        self.assertEqual(R._leakage_from_manifest({})["status"], "not_checked")

    def test_leakage_from_manifest_reads_status(self):
        with tempfile.TemporaryDirectory() as d:
            mp = pathlib.Path(d) / "manifest.json"
            mp.write_text('{"leakage": {"hits": 0, "status": "clean"}}', encoding="utf-8")
            lk = R._leakage_from_manifest({"prepared_manifest": str(mp)})
            self.assertEqual(lk, {"hits": 0, "status": "clean"})


class DeadlineTests(unittest.TestCase):
    def test_should_stop_near_deadline(self):
        self.assertTrue(R.should_stop(time.monotonic() + 1.0, reserve_seconds=30.0))
        self.assertFalse(R.should_stop(time.monotonic() + 120.0, reserve_seconds=30.0))


class RealModeSafeAdapterTests(unittest.TestCase):
    def test_missing_inputs_fail_without_importing_torch(self):
        cfg = _config()
        cfg["runtime"] = {
            "dry_run": False, "train": True,
            "train_pairs": "data/__does_not_exist__/pairs.jsonl",
            "hard_negatives": "data/__does_not_exist__/hn.jsonl",
            "train_base_model": "outputs/__no_such_ckpt__",
            "eval_sets": "webfaq",
        }
        # the safe adapter must not *newly* import torch just to discover missing inputs
        # (a prior torch-based test in the full suite may already have loaded it, so we compare
        # the import state before/after rather than asserting torch is absent outright)
        had_torch = "torch" in sys.modules
        with tempfile.TemporaryDirectory() as d:
            res = R.run_dense_trial(cfg, pathlib.Path(d) / "run",
                                    time.monotonic() + 600, dry_run=False)
            self.assertEqual(res["status"], "fail")
            self.assertIn("missing_inputs", res)
            self.assertTrue(res["missing_inputs"])
        self.assertEqual("torch" in sys.modules, had_torch)


if __name__ == "__main__":
    unittest.main()
