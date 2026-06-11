"""Stdlib tests for the v2 release-gate additions. No ML, no network."""
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import validate_release_2026 as VR  # noqa: E402


def _write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


class TestRerankerPromotion(unittest.TestCase):
    def test_fails_on_germanquad_degradation(self):
        with tempfile.TemporaryDirectory() as d:
            rd = pathlib.Path(d)
            _write(rd / "reranker-lift-germanquad-v2.json",
                   {"first_stage_ndcg@10": 0.886, "student_reranker_ndcg@10": 0.532})
            issues = VR.check_reranker_promotion(rd)
            self.assertTrue(any("germanquad" in i[1] for i in issues))

    def test_passes_when_nonneg(self):
        with tempfile.TemporaryDirectory() as d:
            rd = pathlib.Path(d)
            _write(rd / "reranker-lift-germanquad-v2.json",
                   {"first_stage_ndcg@10": 0.886, "student_reranker_ndcg@10": 0.91})
            _write(rd / "reranker-lift-dt_test-v2.json",
                   {"first_stage_ndcg@10": 0.95, "student_reranker_ndcg@10": 0.99})
            self.assertEqual(VR.check_reranker_promotion(rd), [])


class TestManifestCheck(unittest.TestCase):
    def test_missing_manifest_fails(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(VR.check_v2_manifest(pathlib.Path(d)))

    def test_public_benchmark_trainable_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            _write(root / "configs" / "data_sources_v2.json", {"sources": [{
                "source_id": "leak", "display_name": "x", "source_type": "hf_dataset",
                "domain": "qa_wiki", "license": "CC-BY-4.0", "allowed_for_training": True,
                "public_benchmark": True, "eval_only": True, "notes": "",
                "loader": {"kind": "hf", "path_or_id": "x"}}]})
            issues = VR.check_v2_manifest(root)
            self.assertTrue(any("public_benchmark" in i[1] or "eval_only" in i[1] for i in issues))


class TestV2Artifacts(unittest.TestCase):
    def test_missing_artifacts_fail(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(VR.check_v2_artifacts(pathlib.Path(d)))

    def test_complete_artifacts_pass(self):
        with tempfile.TemporaryDirectory() as d:
            rd = pathlib.Path(d)
            for a in VR.V2_REQUIRED_ARTIFACTS:
                _write(rd / a, {"ok": True})
            self.assertEqual(VR.check_v2_artifacts(rd), [])


class TestFullGate(unittest.TestCase):
    def test_repo_gate_passes(self):
        rep = VR.run_checks(ROOT)
        self.assertEqual(rep["status"], "pass", rep)

    def test_require_v2_artifacts_fails_without_run(self):
        # default results dir (outputs) has no v2 artifacts yet -> release-readiness fails
        rep = VR.run_checks(ROOT, require_v2_artifacts=True)
        self.assertEqual(rep["status"], "fail")
        self.assertTrue(rep["checks"]["v2_artifacts"])


if __name__ == "__main__":
    unittest.main()
