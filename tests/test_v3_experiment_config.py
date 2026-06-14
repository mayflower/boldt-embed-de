"""Stdlib tests for the v3 experiment config loader/validator. No ML deps, no network."""
import copy
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import v3_experiment_config as v3  # noqa: E402

CFG = ROOT / "configs" / "experiments" / "v3_real_domain_generalization.json"


def _write_tmp(obj) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(obj, f)
    f.close()
    return f.name


class TestShippedConfig(unittest.TestCase):
    def test_loads(self):
        c = v3.load_v3_experiment_config(CFG)
        self.assertEqual(c.experiment_id, "v3-real-domain-generalization")
        self.assertTrue(c.public_benchmarks_eval_only)
        self.assertTrue(c.train_only_if_license_known)
        self.assertTrue(c.train_only_if_leakage_full_scan_complete)
        self.assertEqual(c.target_candidate_count_min, 100000)
        self.assertEqual(c.target_teacher_validated_positives_min, 50000)
        self.assertEqual(c.success_criteria["license_unknown_rows_max"], 0)

    def test_fractions_sum_to_one(self):
        c = v3.load_v3_experiment_config(CFG)
        self.assertAlmostEqual(sum(c.domain_fractions().values()), 1.0, places=6)

    def test_target_counts_by_domain(self):
        c = v3.load_v3_experiment_config(CFG)
        counts = c.target_counts_by_domain(100000)
        self.assertEqual(counts["web"], 25000)   # 0.25 * 100000
        self.assertEqual(counts["wiki_non_eval"], 20000)
        self.assertEqual(sum(counts.values()), 100000)

    def test_real_domains_present(self):
        c = v3.load_v3_experiment_config(CFG)
        for dom in ("faq_real", "admin_real", "legal_adjacency_real_no_eval_overlap"):
            self.assertIn(dom, c.domain_targets)

    def test_all_success_criteria_numeric(self):
        c = v3.load_v3_experiment_config(CFG)
        for k, v in c.success_criteria.items():
            self.assertTrue(isinstance(v, (int, float)) and not isinstance(v, bool), k)

    def test_loader_is_stdlib_only(self):
        # Verified in an isolated subprocess (sibling tests import torch into this process).
        import subprocess
        code = ("import sys; sys.path.insert(0, %r);"
                "from boldt_embed import v3_experiment_config as v3;"
                "v3.load_v3_experiment_config(%r);"
                "assert 'torch' not in sys.modules; print('clean')") % (str(ROOT / "src"), str(CFG))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.base = json.loads(CFG.read_text(encoding="utf-8"))

    def test_bad_fractions_fail(self):
        d = copy.deepcopy(self.base)
        d["domain_targets"]["web"] = 0.5  # now sums to > 1.0
        errs = v3.validate_v3_experiment(d)
        self.assertTrue(any("sum to 1.0" in e for e in errs), errs)

    def test_eval_only_false_fails(self):
        d = copy.deepcopy(self.base)
        d["public_benchmarks_eval_only"] = False
        errs = v3.validate_v3_experiment(d)
        self.assertTrue(any("public_benchmarks_eval_only" in e for e in errs), errs)

    def test_license_known_false_fails(self):
        d = copy.deepcopy(self.base)
        d["train_only_if_license_known"] = False
        errs = v3.validate_v3_experiment(d)
        self.assertTrue(any("train_only_if_license_known" in e for e in errs), errs)

    def test_leakage_full_scan_false_fails(self):
        d = copy.deepcopy(self.base)
        d["train_only_if_leakage_full_scan_complete"] = False
        errs = v3.validate_v3_experiment(d)
        self.assertTrue(any("train_only_if_leakage_full_scan_complete" in e for e in errs), errs)

    def test_license_unknown_rows_max_nonzero_fails(self):
        d = copy.deepcopy(self.base)
        d["success_criteria"]["license_unknown_rows_max"] = 5
        errs = v3.validate_v3_experiment(d)
        self.assertTrue(any("license_unknown_rows_max MUST be 0" in e for e in errs), errs)

    def test_missing_license_unknown_rows_max_fails(self):
        d = copy.deepcopy(self.base)
        del d["success_criteria"]["license_unknown_rows_max"]
        errs = v3.validate_v3_experiment(d)
        self.assertTrue(any("license_unknown_rows_max is required" in e for e in errs), errs)

    def test_nonnumeric_success_criterion_fails(self):
        d = copy.deepcopy(self.base)
        d["success_criteria"]["dense_gerdalir_ndcg10_min"] = "high"
        errs = v3.validate_v3_experiment(d)
        self.assertTrue(any("success_criteria" in e for e in errs), errs)

    def test_validated_positives_above_candidates_fails(self):
        d = copy.deepcopy(self.base)
        d["target_teacher_validated_positives_min"] = 200000  # > candidate min
        errs = v3.validate_v3_experiment(d)
        self.assertTrue(any("must be <= target_candidate_count_min" in e for e in errs), errs)

    def test_negative_count_fails(self):
        d = copy.deepcopy(self.base)
        d["target_candidate_count_min"] = -1
        errs = v3.validate_v3_experiment(d)
        self.assertTrue(any("target_candidate_count_min" in e for e in errs), errs)

    def test_shipped_config_has_no_errors(self):
        self.assertEqual(v3.validate_v3_experiment(self.base), [])

    def test_load_raises_on_invalid(self):
        d = copy.deepcopy(self.base)
        d["train_only_if_license_known"] = False
        with self.assertRaises(ValueError):
            v3.load_v3_experiment_config(_write_tmp(d))


if __name__ == "__main__":
    unittest.main()
