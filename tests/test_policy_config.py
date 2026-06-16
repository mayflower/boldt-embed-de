"""Stdlib tests for the rerank-policy artifact loader/validator. No ML."""
import copy
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import policy_config as PC  # noqa: E402

POLICY = ROOT / "configs" / "policies" / "bounded_margin_override_v1.json"


def _policy():
    return json.loads(POLICY.read_text("utf-8"))


class TestShippedPolicy(unittest.TestCase):
    def test_valid_policy_passes(self):
        self.assertEqual(PC.validate_policy(_policy()), [])

    def test_loads(self):
        d = PC.load_policy(POLICY)
        self.assertEqual(d["policy_id"], "bounded_margin_override_v1")
        self.assertFalse(d["raw_always_rerank_recommended"])
        self.assertEqual(d["recommended_mode"], "policy_gated_only")

    def test_shipped_model_checkpoint_present(self):
        # the pinned original conservative checkpoint exists on disk
        ok, _ = PC.check_model_exists(_policy(), root=ROOT, require=True)
        self.assertTrue(ok)


class TestGuards(unittest.TestCase):
    def test_raw_recommended_fails(self):
        d = _policy(); d["raw_always_rerank_recommended"] = True
        self.assertTrue(any("raw_always_rerank_recommended" in e for e in PC.validate_policy(d)))

    def test_raw_recommended_mode_fails(self):
        d = _policy(); d["recommended_mode"] = "raw_always_rerank"
        self.assertTrue(any("recommended_mode" in e for e in PC.validate_policy(d)))

    def test_qrels_as_inference_feature_fails(self):
        d = _policy(); d["features_allowed_at_inference"] = d["features_allowed_at_inference"] + ["qrels"]
        self.assertTrue(any("overlap" in e for e in PC.validate_policy(d)))

    def test_missing_bounds_fails(self):
        d = _policy(); d.pop("bounds")
        self.assertTrue(any("bounds" in e for e in PC.validate_policy(d)))

    def test_missing_validation_thresholds_fails(self):
        d = _policy(); d["validation"].pop("max_germanquad_catastrophic")
        self.assertTrue(any("max_germanquad_catastrophic" in e for e in PC.validate_policy(d)))

    def test_missing_model_checkpoint_fails(self):
        d = _policy(); d["model_checkpoint"] = ""
        self.assertTrue(any("model_checkpoint" in e for e in PC.validate_policy(d)))

    def test_load_raises_on_invalid(self):
        d = _policy(); d["raw_always_rerank_recommended"] = True
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(d, f); f.close()
        with self.assertRaises(ValueError):
            PC.load_policy(f.name)


class TestModelExistence(unittest.TestCase):
    def test_missing_path_warns_in_dry_run_fails_when_required(self):
        d = _policy(); d["model_checkpoint"] = "outputs/does/not/exist"
        ok_dry, msg_dry = PC.check_model_exists(d, root=ROOT, require=False)
        self.assertTrue(ok_dry)                       # dry-run: warning only
        self.assertTrue(msg_dry.startswith("WARNING"))
        ok_req, _ = PC.check_model_exists(d, root=ROOT, require=True)
        self.assertFalse(ok_req)                      # --require-model-exists: failure


if __name__ == "__main__":
    unittest.main()
