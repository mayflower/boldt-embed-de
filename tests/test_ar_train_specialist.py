"""P07 — specialist trainer: catalogue + warm-start validation, dry-run plan, manifest (stdlib)."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("ar_train_specialist",
                                               ROOT / "scripts" / "ar_train_specialist.py")
ats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ats)

FAKE_CAT = {
    "wiki": {"id": "wiki", "training_usable": True, "leakage": "scanned_clean", "rows": 1000},
    "bad_usable": {"id": "bad_usable", "training_usable": False, "leakage": "scanned_clean"},
    "bad_leak": {"id": "bad_leak", "training_usable": True, "leakage": "unscanned"},
}

# warm_start is a non-local ref (HF id) so the test never depends on a checkpoint on disk
SPEC = {
    "name": "t_specialists", "warm_start": "Boldt/Boldt-DC-350M", "default_steps": 6000,
    "sources": [{"id": "wiki", "label": "wiki_miracl", "steps": 4000, "weight": 1.0},
                {"id": "bad_usable", "label": "bu"}, {"id": "bad_leak", "label": "bl"}],
    "training": {"batch_size": 32, "grad_accumulation": 8, "learning_rate": 1e-5},
}


class TestPlanSpecialist(unittest.TestCase):
    def test_no_torch(self):
        import pathlib as _p
        sys.path.insert(0, str(_p.Path(__file__).resolve().parent))
        from torch_free import script_is_torch_free
        self.assertTrue(script_is_torch_free("scripts/ar_train_specialist.py"))

    def test_valid_source_plan(self):
        p = ats.plan_specialist(SPEC, "wiki", "outputs/v8/specialists", catalogue=FAKE_CAT)
        self.assertEqual(p["errors"], [])
        self.assertEqual(p["run_id"], "spec-wiki_miracl")
        self.assertEqual(p["experiment_config"]["data_mixture"], {"wiki": 1.0})
        rt = p["experiment_config"]["runtime"]
        self.assertTrue(rt["materialize_mixture"])
        self.assertEqual(rt["train_base_model"], "Boldt/Boldt-DC-350M")
        self.assertEqual(p["experiment_config"]["training"]["max_steps"], 4000)
        self.assertIn("ar_loop.py", p["command"])
        for f in ("--real", "--allow-gpu", "--allow-checkpoints"):
            self.assertIn(f, p["command"])
        self.assertEqual(p["manifest"]["source_id"], "wiki")
        self.assertEqual(p["manifest"]["leakage"]["status"], "scanned_clean")

    def test_unknown_source_id_in_config_fails(self):
        p = ats.plan_specialist(SPEC, "not_listed", "out", catalogue=FAKE_CAT)
        self.assertTrue(any("not listed" in e for e in p["errors"]))

    def test_source_not_in_catalogue_fails(self):
        spec = dict(SPEC, sources=[{"id": "ghost", "label": "g"}])
        p = ats.plan_specialist(spec, "ghost", "out", catalogue=FAKE_CAT)
        self.assertTrue(any("not in configs/data_sources.json" in e for e in p["errors"]))

    def test_training_usable_false_fails(self):
        p = ats.plan_specialist(SPEC, "bad_usable", "out", catalogue=FAKE_CAT)
        self.assertTrue(any("training_usable=false" in e for e in p["errors"]))

    def test_unscanned_leakage_fails(self):
        p = ats.plan_specialist(SPEC, "bad_leak", "out", catalogue=FAKE_CAT)
        self.assertTrue(any("leakage" in e for e in p["errors"]))

    def test_missing_local_warm_start_fails(self):
        spec = dict(SPEC, warm_start="outputs/does/not/exist/checkpoint")
        p = ats.plan_specialist(spec, "wiki", "out", catalogue=FAKE_CAT)
        self.assertTrue(any("warm_start" in e for e in p["errors"]))

    def test_dry_run_main_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "spec.json"
            cfg.write_text(json.dumps(SPEC), encoding="utf-8")
            # patch the catalogue loader so the test is hermetic
            orig = ats.recipe._load_catalogue
            ats.recipe._load_catalogue = lambda: FAKE_CAT
            try:
                rc = ats.main(["--config", str(cfg), "--source-id", "wiki",
                               "--out-root", f"{d}/sp", "--dry-run"])
            finally:
                ats.recipe._load_catalogue = orig
            self.assertEqual(rc, 0)
            run_dir = Path(d) / "sp" / "spec-wiki_miracl"
            self.assertTrue((run_dir / "experiment.json").exists())
            self.assertTrue((run_dir / "specialist_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
