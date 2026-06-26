"""Tests for the merge-search orchestrator (Prompt 08) — stdlib only.

Covers: dry-run produces the planned-candidate JSON, dry-run imports NO torch, and base-relative
methods (task_vector_sum / ties / dare_linear) without a warm_start are reported unsupported.
"""
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import ar_merge_search  # noqa: E402


BASE_CONFIG = {
    "name": "v8_merge_search",
    "parents": [
        {"label": "wiki_miracl", "path": "outputs/v8/specialists/spec-wiki_miracl/checkpoint"},
        {"label": "web_msmarco", "path": "outputs/v8/specialists/spec-web_msmarco/checkpoint"},
        {"label": "faq_web", "path": "outputs/v8/specialists/spec-faq_web/checkpoint"},
    ],
    "warm_start": "outputs/v8/diverse-causal/checkpoint",
    "methods": [
        {"name": "mean"},
        {"name": "weighted_mean", "weights_grid": [[0.5, 0.25, 0.25], [0.25, 0.5, 0.25]]},
        {"name": "ties", "density_grid": [0.2, 0.5]},
        {"name": "dare_linear", "density_grid": [0.2, 0.5], "rescale": True},
    ],
}


class TestDryRunPlanning(unittest.TestCase):
    def test_dry_run_no_torch_imported(self):
        # Importing the orchestrator + planning must not pull torch in — verified in a fresh
        # subprocess so discover's torch-using modules can't pollute the shared sys.modules.
        sys.path.insert(0, str(ROOT / "tests"))
        from torch_free import script_is_torch_free
        self.assertTrue(script_is_torch_free("scripts/ar_merge_search.py"))
        report = ar_merge_search.build_dry_run_report(BASE_CONFIG, "outputs/merged/test")
        self.assertTrue(report["dry_run"])

    def test_dry_run_report_is_valid_json_with_candidates(self):
        report = ar_merge_search.build_dry_run_report(BASE_CONFIG, "outputs/merged/test")
        # round-trips as JSON
        round_tripped = json.loads(json.dumps(report))
        self.assertEqual(round_tripped["n_parents"], 3)
        cands = report["planned_candidates"]
        self.assertTrue(len(cands) > 0)
        methods = [c["method"] for c in cands]
        self.assertIn("mean", methods)
        self.assertIn("weighted_mean", methods)
        self.assertIn("ties", methods)
        self.assertIn("dare_linear", methods)
        # grid expansion: 2 weight rows + 2 ties densities + 2 dare densities + 1 mean = 7
        self.assertEqual(len(cands), 7)
        # every candidate carries a label + params block
        for c in cands:
            self.assertIn("label", c)
            self.assertIn("params", c)

    def test_base_relative_unsupported_without_warm_start(self):
        cfg = dict(BASE_CONFIG)
        cfg.pop("warm_start", None)
        cfg = {**cfg, "warm_start": None}
        supported, unsupported = ar_merge_search.plan_candidates(cfg)
        unsupported_methods = {u["method"] for u in unsupported}
        self.assertIn("ties", unsupported_methods)
        self.assertIn("dare_linear", unsupported_methods)
        for u in unsupported:
            if u["method"] in ("ties", "dare_linear", "task_vector_sum"):
                self.assertIn("warm_start", u["reason"])
        # the base-free methods still plan fine
        supported_methods = {c["method"] for c in supported}
        self.assertIn("mean", supported_methods)
        self.assertIn("weighted_mean", supported_methods)

    def test_task_vector_sum_without_warm_start_unsupported(self):
        cfg = {
            "name": "x",
            "parents": BASE_CONFIG["parents"],
            "warm_start": None,
            "methods": [{"name": "task_vector_sum"}],
        }
        supported, unsupported = ar_merge_search.plan_candidates(cfg)
        self.assertEqual(supported, [])
        self.assertEqual(len(unsupported), 1)
        self.assertEqual(unsupported[0]["method"], "task_vector_sum")

    def test_slerp_requires_exactly_two_parents(self):
        cfg = {
            "name": "x",
            "parents": BASE_CONFIG["parents"],  # 3 parents
            "warm_start": "w",
            "methods": [{"name": "slerp_pairwise", "t_grid": [0.5]}],
        }
        supported, unsupported = ar_merge_search.plan_candidates(cfg)
        self.assertEqual(supported, [])
        self.assertEqual(unsupported[0]["method"], "slerp_pairwise")

        cfg2 = {
            "name": "x",
            "parents": BASE_CONFIG["parents"][:2],  # exactly 2
            "warm_start": "w",
            "methods": [{"name": "slerp_pairwise", "t_grid": [0.3, 0.7]}],
        }
        supported2, _ = ar_merge_search.plan_candidates(cfg2)
        self.assertEqual(len(supported2), 2)

    def test_weighted_mean_wrong_length_unsupported(self):
        cfg = {
            "name": "x",
            "parents": BASE_CONFIG["parents"],  # 3 parents
            "methods": [{"name": "weighted_mean", "weights_grid": [[0.5, 0.5]]}],  # only 2 weights
        }
        supported, unsupported = ar_merge_search.plan_candidates(cfg)
        self.assertEqual(supported, [])
        self.assertTrue(any(u["method"] == "weighted_mean" for u in unsupported))

    def test_unknown_method_unsupported(self):
        cfg = {"name": "x", "parents": BASE_CONFIG["parents"], "methods": [{"name": "bogus"}]}
        _, unsupported = ar_merge_search.plan_candidates(cfg)
        self.assertEqual(unsupported[0]["method"], "bogus")

    def test_shipped_config_dry_run_main_no_torch(self):
        # Run the actual CLI entrypoint on the shipped config IN A FRESH SUBPROCESS and assert it
        # neither errors nor imports torch (order-independent, unlike a shared sys.modules check).
        sys.path.insert(0, str(ROOT / "tests"))
        from torch_free import is_torch_free
        snippet = (
            "import importlib.util as u\n"
            "s = u.spec_from_file_location('m', 'scripts/ar_merge_search.py')\n"
            "m = u.module_from_spec(s); s.loader.exec_module(m)\n"
            "rc = m.main(['--config', 'configs/autoresearch/merge_search_v8.json',"
            " '--out', 'outputs/merged/v8_merge_search', '--dry-run'])\n"
            "assert rc == 0, rc")
        self.assertTrue(is_torch_free(snippet))


if __name__ == "__main__":
    unittest.main()
