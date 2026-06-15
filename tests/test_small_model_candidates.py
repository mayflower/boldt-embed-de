"""Stdlib tests for the v5 small-model candidate comparison + selection gate. No ML, no network."""
import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import small_model_candidates as SMC  # noqa: E402

CFG = ROOT / "configs" / "v5_small_model_candidates.json"


def _cfg():
    return json.loads(CFG.read_text("utf-8"))


def res(name, family, quality, latency_ms, harness="h1", **kw):
    r = {"name": name, "family": family, "quality": quality, "latency_ms": latency_ms,
         "harness": harness}
    r.update(kw)
    return r


class TestConfig(unittest.TestCase):
    def test_shipped_config_valid(self):
        self.assertEqual(SMC.validate_candidates_config(_cfg()), [])

    def test_includes_boldt_and_qwen(self):
        d = _cfg()
        dense_fams = {c["family"] for c in d["dense_candidates"]}
        rer_fams = {c["family"] for c in d["reranker_candidates"]}
        self.assertIn("boldt", dense_fams); self.assertIn("qwen3", dense_fams)
        self.assertIn("boldt", rer_fams); self.assertIn("qwen3", rer_fams)

    def test_single_family_rejected(self):
        d = _cfg()
        d["reranker_candidates"] = [d["reranker_candidates"][0]]   # boldt only
        self.assertTrue(any("model families" in e for e in SMC.validate_candidates_config(d)))

    def test_full_finetune_requires_flag(self):
        d = _cfg(); d["tuning"]["method"] = "full"; d["tuning"]["full_finetune_allowed"] = False
        self.assertTrue(any("full_finetune_allowed" in e for e in SMC.validate_candidates_config(d)))

    def test_missing_teacher_rejected(self):
        d = _cfg(); d["teachers"].pop("reranker")
        self.assertTrue(any("teachers.reranker" in e for e in SMC.validate_candidates_config(d)))

    def test_bad_latency_rejected(self):
        d = _cfg(); d["selection_gate"]["max_reranker_latency_ms"] = 0
        self.assertTrue(any("max_reranker_latency_ms" in e for e in SMC.validate_candidates_config(d)))


class TestSelectionGate(unittest.TestCase):
    def test_higher_quality_wins_when_outside_tie_band(self):
        out = SMC.select_default([res("boldt", "boldt", 0.90, 50), res("qwen", "qwen3", 0.80, 10)],
                                 max_latency_ms=100, tie_break_quality_delta=0.005)
        self.assertEqual(out["default"], "boldt")

    def test_tie_band_prefers_faster(self):
        out = SMC.select_default([res("boldt", "boldt", 0.800, 50), res("qwen", "qwen3", 0.802, 10)],
                                 max_latency_ms=100, tie_break_quality_delta=0.005)
        self.assertEqual(out["default"], "qwen")        # within 0.005 quality -> faster wins

    def test_selection_is_family_blind(self):
        # same numbers, families swapped -> the faster model is chosen either way
        a = SMC.select_default([res("m1", "boldt", 0.800, 50), res("m2", "qwen3", 0.802, 10)],
                               max_latency_ms=100)
        b = SMC.select_default([res("m1", "qwen3", 0.800, 50), res("m2", "boldt", 0.802, 10)],
                               max_latency_ms=100)
        self.assertEqual(a["default"], "m2")
        self.assertEqual(b["default"], "m2")            # family label change does not move the pick

    def test_latency_budget_excludes(self):
        out = SMC.select_default([res("slow", "boldt", 0.95, 200), res("fast", "qwen3", 0.80, 10)],
                                 max_latency_ms=50)
        self.assertEqual(out["default"], "fast")        # slow excluded despite higher quality
        self.assertTrue(any(x["name"] == "slow" for x in out["excluded"]))

    def test_256d_retention_gate_for_dense(self):
        out = SMC.select_default(
            [res("a", "boldt", 0.90, 10, retention_256d=0.90),
             res("b", "qwen3", 0.85, 10, retention_256d=0.97)],
            max_latency_ms=50, min_256d_retention=0.95)
        self.assertEqual(out["default"], "b")           # a fails 256d retention

    def test_requires_two_candidates_same_harness(self):
        self.assertEqual(SMC.select_default([res("only", "boldt", 0.9, 10)],
                                            max_latency_ms=50)["status"], "insufficient_comparison")

    def test_rejects_inconsistent_harness(self):
        out = SMC.select_default([res("a", "boldt", 0.9, 10, harness="h1"),
                                  res("b", "qwen3", 0.8, 10, harness="h2")], max_latency_ms=50)
        self.assertEqual(out["status"], "inconsistent_harness")


class TestHelpers(unittest.TestCase):
    def test_storage_table(self):
        t = SMC.storage_table([1024, 256])
        self.assertEqual(t["256"], {"fp32_bytes": 1024, "fp16_bytes": 512})

    def test_measurement_plan(self):
        plan = SMC.measurement_plan(_cfg(), tune_reranker=True, tune_embedding=False,
                                    full_finetune=False)
        self.assertEqual(plan["tuning"]["method"], "lora")
        self.assertTrue(plan["tuning"]["reranker_lora"])
        self.assertIn("quality", plan["reported_metrics"])
        self.assertIn("latency_ms", plan["reported_metrics"])


class TestDryRunNoMl(unittest.TestCase):
    def test_cli_dry_run_no_ml(self):
        with tempfile.TemporaryDirectory() as d:
            report = pathlib.Path(d) / "rep.json"
            code = (
                "import sys; sys.path.insert(0, %r); sys.argv=['x','--config', %r, '--report', %r, "
                "'--dry-run']\n"
                "import runpy; "
                "rc=0\n"
                "try:\n runpy.run_path(%r, run_name='__main__')\n"
                "except SystemExit as e:\n rc=e.code or 0\n"
                "import sys as _s; assert 'torch' not in _s.modules, 'torch imported'\n"
                "print('RC', rc)\n"
                % (str(ROOT / "src"), str(CFG), str(report),
                   str(ROOT / "scripts" / "eval_small_model_candidates.py"))
            )
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("RC 0", r.stdout)
            self.assertTrue(report.exists())
            rep = json.loads(report.read_text("utf-8"))
            self.assertIn("plan", rep)
            self.assertIn("boldt-causal-v3", rep["plan"]["dense_candidates"])


if __name__ == "__main__":
    unittest.main()
