"""Stdlib tests for the v2 experiment config loader/validator. No ML deps, no network."""
import copy
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import v2_experiment_config as v2  # noqa: E402

CFG = ROOT / "configs" / "experiments" / "v2_generalization.json"


def _write_tmp(obj) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(obj, f)
    f.close()
    return f.name


class TestShippedConfig(unittest.TestCase):
    def test_loads(self):
        c = v2.load_v2_experiment_config(CFG)
        self.assertEqual(c.experiment_id, "v2-data-scale-generalization")
        self.assertEqual(c.target_candidate_count_min, 50000)
        self.assertEqual(c.target_candidate_count_stretch, 250000)
        self.assertTrue(c.public_benchmarks_eval_only)
        self.assertIn("GerDaLIR", c.held_out_eval_sets)
        self.assertEqual(c.reranker["negatives_per_query"], 8)

    def test_fractions_sum_to_one(self):
        c = v2.load_v2_experiment_config(CFG)
        self.assertAlmostEqual(sum(c.domain_fractions().values()), 1.0, places=6)

    def test_target_counts_by_domain(self):
        c = v2.load_v2_experiment_config(CFG)
        counts = c.target_counts_by_domain(50000)
        self.assertEqual(counts["web"], 12500)   # 0.25 * 50000
        self.assertEqual(sum(counts.values()), 50000)

    def test_loader_is_stdlib_only(self):
        # Verified in an isolated subprocess (sibling tests import torch into this process).
        import subprocess
        code = ("import sys; sys.path.insert(0, %r);"
                "from boldt_embed import v2_experiment_config as v2;"
                "v2.load_v2_experiment_config(%r);"
                "assert 'torch' not in sys.modules; print('clean')") % (str(ROOT / "src"), str(CFG))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.base = json.loads(CFG.read_text(encoding="utf-8"))

    def test_bad_fractions_fail(self):
        d = copy.deepcopy(self.base)
        d["domains"]["web"]["target_fraction"] = 0.5  # now sums to 1.25
        errs = v2.validate_v2_experiment(d)
        self.assertTrue(any("sum to 1.0" in e for e in errs), errs)

    def test_eval_only_false_fails(self):
        d = copy.deepcopy(self.base)
        d["public_benchmarks_eval_only"] = False
        errs = v2.validate_v2_experiment(d)
        self.assertTrue(any("public_benchmarks_eval_only" in e for e in errs), errs)

    def test_stretch_below_min_fails(self):
        d = copy.deepcopy(self.base)
        d["target_candidate_count_stretch"] = 10
        errs = v2.validate_v2_experiment(d)
        self.assertTrue(any("stretch must be >=" in e for e in errs), errs)

    def test_nonnumeric_success_criterion_fails(self):
        d = copy.deepcopy(self.base)
        d["success_criteria"]["dense_germanquad_ndcg10_min"] = "high"
        errs = v2.validate_v2_experiment(d)
        self.assertTrue(any("success_criteria" in e for e in errs), errs)

    def test_negative_count_fails(self):
        d = copy.deepcopy(self.base)
        d["target_candidate_count_min"] = -1
        errs = v2.validate_v2_experiment(d)
        self.assertTrue(any("target_candidate_count_min" in e for e in errs), errs)

    def test_load_raises_on_invalid(self):
        d = copy.deepcopy(self.base)
        d["public_benchmarks_eval_only"] = False
        with self.assertRaises(ValueError):
            v2.load_v2_experiment_config(_write_tmp(d))


if __name__ == "__main__":
    unittest.main()
