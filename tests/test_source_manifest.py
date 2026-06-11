"""Stdlib tests for the v2 source manifest (load/validate, fail-closed rules). No network."""
import copy
import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import source_manifest as sm  # noqa: E402

SHIPPED = ROOT / "configs" / "data_sources_v2.json"
FIXTURE = ROOT / "tests" / "fixtures" / "v2_sources_manifest.json"
SCRIPT = ROOT / "scripts" / "validate_data_sources_v2.py"


class TestShippedManifest(unittest.TestCase):
    def test_loads_and_valid(self):
        entries = sm.load_source_manifest(SHIPPED)
        self.assertGreaterEqual(len(entries), 10)

    def test_public_benchmarks_blocked_from_training(self):
        entries = sm.load_source_manifest(SHIPPED)
        for e in entries:
            if e.public_benchmark or e.eval_only:
                self.assertFalse(e.allowed_for_training, f"{e.source_id} must not train")

    def test_known_eval_sets_present_and_eval_only(self):
        ids = {e.source_id: e for e in sm.load_source_manifest(SHIPPED)}
        for sid in ("germanquad", "gerdalir"):
            self.assertIn(sid, ids)
            self.assertTrue(ids[sid].eval_only)
            self.assertFalse(ids[sid].allowed_for_training)

    def test_training_sources_use_training_domains(self):
        for e in sm.training_sources(sm.load_source_manifest(SHIPPED)):
            self.assertIn(e.domain, sm.TRAINING_DOMAINS, e.source_id)

    def test_future_sources_blocked(self):
        ids = {e.source_id: e for e in sm.load_source_manifest(SHIPPED)}
        self.assertFalse(ids["mmarco_de"].allowed_for_training)   # uncertain license
        self.assertFalse(ids["clips_mqa_de"].allowed_for_training)


class TestValidationRules(unittest.TestCase):
    def setUp(self):
        self.base = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_fixture_valid(self):
        self.assertEqual(sm.validate_source_manifest(self.base), [])

    def test_public_benchmark_training_allowed_fails(self):
        d = copy.deepcopy(self.base)
        d["sources"][1]["allowed_for_training"] = True  # eval/public -> can't train
        errs = sm.validate_source_manifest(d)
        self.assertTrue(any("public_benchmark" in e or "eval_only" in e for e in errs), errs)

    def test_missing_license_fails(self):
        d = copy.deepcopy(self.base)
        d["sources"][0]["license"] = ""
        errs = sm.validate_source_manifest(d)
        self.assertTrue(any("missing license" in e for e in errs), errs)

    def test_uncertain_license_blocks_training(self):
        d = copy.deepcopy(self.base)
        d["sources"][0]["license"] = "uncertain-verify"
        errs = sm.validate_source_manifest(d)
        self.assertTrue(any("uncertain license" in e for e in errs), errs)

    def test_unknown_domain_fails(self):
        d = copy.deepcopy(self.base)
        d["sources"][0]["domain"] = "banana"
        errs = sm.validate_source_manifest(d)
        self.assertTrue(any("unknown domain" in e for e in errs), errs)

    def test_eval_domain_not_trainable(self):
        d = copy.deepcopy(self.base)
        d["sources"][0]["domain"] = "legal"  # eval content domain, but training-allowed
        errs = sm.validate_source_manifest(d)
        self.assertTrue(any("training domain" in e for e in errs), errs)


class TestCLI(unittest.TestCase):
    def test_cli_valid_manifest(self):
        out = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(SHIPPED),
                              "--format", "json"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn('"status": "ok"', out.stdout)

    def test_cli_fails_on_bad(self):
        import tempfile
        d = json.loads(FIXTURE.read_text())
        d["sources"][1]["allowed_for_training"] = True
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(d, f); f.close()
        out = subprocess.run([sys.executable, str(SCRIPT), "--manifest", f.name],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 1)
        self.assertIn("INVALID", out.stderr)


if __name__ == "__main__":
    unittest.main()
