"""P13 — hybrid/multivector ceiling-breaker stub CLIs: dry-run plan + fail-closed (stdlib)."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mvp = _load("ar_multivector_plan")
hyb = _load("ar_hybrid_eval")
TRACK = json.loads((ROOT / "configs" / "autoresearch" / "hybrid_track.json").read_text())


class TestStubs(unittest.TestCase):
    def test_no_torch(self):
        import pathlib as _p
        sys.path.insert(0, str(_p.Path(__file__).resolve().parent))
        from torch_free import script_is_torch_free
        self.assertTrue(script_is_torch_free("scripts/ar_multivector_plan.py"))
        self.assertTrue(script_is_torch_free("scripts/ar_hybrid_eval.py"))

    def test_track_config_present(self):
        self.assertFalse(TRACK["is_product_default"])
        self.assertTrue(TRACK["rules"]["dense_single_vector_remains_primary"])

    def test_multivector_plan_known_mode(self):
        p = mvp.plan_mode("colbert_late_interaction", TRACK)
        self.assertEqual(p["errors"], [])
        self.assertTrue(p["planned_steps"])
        self.assertFalse(p["is_product_default"])

    def test_multivector_plan_unknown_mode_fails(self):
        p = mvp.plan_mode("nope", TRACK)
        self.assertTrue(p["errors"])

    def test_hybrid_eval_missing_model_fails(self):
        p = hyb.plan_eval("outputs/does/not/exist", "reranked_two_stage", ["GermanQuAD-Retrieval"])
        self.assertTrue(any("does not exist" in e for e in p["errors"]))

    def test_hybrid_eval_remote_model_ok(self):
        p = hyb.plan_eval("Boldt/Boldt-DC-350M", "reranked_two_stage", ["GermanQuAD-Retrieval"])
        self.assertEqual(p["errors"], [])

    def test_multivector_main_dry_run(self):
        rc = mvp.main(["--mode", "sparse_dense_hybrid", "--dry-run"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
