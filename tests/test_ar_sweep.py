"""Tests for the AutoResearch sweep driver (stdlib: grid + config generation + dry-run plan)."""
import contextlib
import importlib.util
import io
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load("ar_sweep")


class GridTests(unittest.TestCase):
    def test_grid_size_and_fields(self):
        grid = S.make_grid(20)
        self.assertEqual(len(grid), 20)
        for g in grid:
            self.assertIn(g["temperature"], S.TEMPS)
            self.assertIn(g["learning_rate"], S.LRS)
            self.assertIn(g["warmup_ratio"], S.WARMUPS)
        self.assertEqual(grid[0]["run_id"], "sweep-01")

    def test_config_uses_clean_data_and_knobs(self):
        g = S.make_grid(3)[2]
        cfg = S.config_for(g, max_steps=400)
        self.assertEqual(cfg["extends"], "configs/autoresearch/base_dense.json")
        self.assertEqual(cfg["loss"]["temperature"], g["temperature"])
        self.assertEqual(cfg["training"]["learning_rate"], g["learning_rate"])
        self.assertEqual(cfg["training"]["max_steps"], 400)
        # MUST train on the leakage-clean data, not the raw (leaky) base default
        self.assertEqual(cfg["runtime"]["train_pairs"], S.CLEAN_TRAIN_PAIRS)
        self.assertTrue(cfg["runtime"]["write_checkpoints"])


class DryRunTests(unittest.TestCase):
    def test_dry_run_prints_grid_no_ml(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = S.main(["--dry-run", "--loops", "20", "--max-steps", "400"])
        self.assertEqual(rc, 0)
        d = json.loads(buf.getvalue())
        self.assertEqual(d["loops"], 20)
        self.assertEqual(len(d["grid"]), 20)


if __name__ == "__main__":
    unittest.main()
