"""P02 — catalogue-aware validation of a materialized data_mixture (stdlib only).

`validate_recipe_config` must, when ``runtime.materialize_mixture`` is true, reject a mixture whose
source ids are unknown / training_usable:false / not leakage-clean, and accept a scanned_clean mix.
The checks are gated on materialize_mixture so a non-materialized (dry-run pseudo-metric) mix is
unaffected. See docs/autoresearch-implementation-plan-2026.md (Prompt 02)."""
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import autoresearch_recipe as r  # noqa: E402

# A tiny controlled catalogue so the test never depends on the real configs/data_sources.json.
FAKE_CATALOGUE = {
    "good_wiki": {"id": "good_wiki", "training_usable": True, "leakage": "scanned_clean"},
    "good_clean": {"id": "good_clean", "training_usable": True, "leakage": "clean"},
    "not_usable": {"id": "not_usable", "training_usable": False, "leakage": "scanned_clean"},
    "unscanned": {"id": "unscanned", "training_usable": True, "leakage": "unscanned"},
    "eval_adj": {"id": "eval_adj", "training_usable": False, "leakage": "eval_adjacent"},
}


def _cfg(mix, materialize=True):
    return {
        "task": "dense_retriever",
        "data_mixture": mix,
        "runtime": {"materialize_mixture": materialize},
    }


class TestMaterializedMixtureValidation(unittest.TestCase):
    def setUp(self):
        self._patch = mock.patch.object(r, "_load_catalogue", return_value=FAKE_CATALOGUE)
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def test_no_torch_at_import(self):
        import pathlib as _p
        sys.path.insert(0, str(_p.Path(__file__).resolve().parent))
        from torch_free import module_is_torch_free
        self.assertTrue(module_is_torch_free("boldt_embed.autoresearch_recipe"))

    def test_valid_scanned_clean_mix_passes(self):
        errs = r.validate_recipe_config(_cfg({"good_wiki": 0.5, "good_clean": 0.5}))
        self.assertEqual(errs, [], errs)

    def test_unknown_source_fails_and_names_it(self):
        errs = r.validate_recipe_config(_cfg({"good_wiki": 0.5, "nope": 0.5}))
        self.assertTrue(any("'nope'" in e and "not in configs/data_sources.json" in e for e in errs), errs)

    def test_training_usable_false_fails(self):
        errs = r.validate_recipe_config(_cfg({"not_usable": 1.0}))
        self.assertTrue(any("'not_usable'" in e and "training_usable=false" in e for e in errs), errs)

    def test_unscanned_fails_when_materialized(self):
        errs = r.validate_recipe_config(_cfg({"unscanned": 1.0}))
        self.assertTrue(any("'unscanned'" in e and "leakage" in e for e in errs), errs)

    def test_eval_adjacent_fails(self):
        errs = r.validate_recipe_config(_cfg({"eval_adj": 1.0}))
        self.assertTrue(any("'eval_adj'" in e for e in errs), errs)

    def test_zero_weight_fails_when_materialized(self):
        # weights must sum to 1.0, so pair the 0 with a 1.0 to isolate the >0 rule
        errs = r.validate_recipe_config(_cfg({"good_wiki": 1.0, "good_clean": 0.0}))
        self.assertTrue(any("'good_clean'" in e and "> 0" in e for e in errs), errs)

    def test_not_materialized_skips_catalogue_check(self):
        # same unknown id, but materialize_mixture=false -> catalogue is NOT consulted
        errs = r.validate_recipe_config(_cfg({"nope": 1.0}, materialize=False))
        self.assertFalse(any("data_sources.json" in e for e in errs), errs)


class TestRealCatalogueAndBaseConfig(unittest.TestCase):
    """Integration sanity: the real catalogue + the cleaned base_dense.json mixture."""

    def test_base_dense_mixture_is_catalogue_clean(self):
        import json
        base = json.loads((ROOT / "configs" / "autoresearch" / "base_dense.json").read_text())
        cat = r._load_catalogue()
        for sid in base["data_mixture"]:
            self.assertIn(sid, cat, f"base_dense mixture id {sid!r} missing from catalogue")
            self.assertTrue(cat[sid].get("training_usable"), sid)
            self.assertIn(cat[sid].get("leakage"), ("scanned_clean", "clean"), sid)

    def test_real_unscanned_source_rejected_when_materialized(self):
        # german_stress is training_usable but leakage=unscanned in the real catalogue
        errs = r.validate_recipe_config(_cfg({"german_stress": 1.0}))
        self.assertTrue(any("german_stress" in e for e in errs), errs)


if __name__ == "__main__":
    unittest.main()
