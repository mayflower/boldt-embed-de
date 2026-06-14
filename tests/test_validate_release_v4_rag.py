"""Tests for the v4 RAG release-gate additions. Pure stdlib."""
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import validate_release_2026 as VR  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "v4_rag_results"


class TestV4Artifacts(unittest.TestCase):
    def test_fixture_artifacts_present(self):
        self.assertEqual(VR.check_v4_rag_artifacts(VR.ROOT, FIX), [])

    def test_missing_artifacts_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            issues = VR.check_v4_rag_artifacts(VR.ROOT, pathlib.Path(d))
            self.assertTrue(any(i[0] == "missing_v4_rag_artifact" for i in issues))


class TestV4Card(unittest.TestCase):
    def _card(self, d, text):
        cd = pathlib.Path(d) / "model_cards"
        cd.mkdir(parents=True)
        (cd / "Boldt-Reranker-DE-350M-v1.md").write_text(text, encoding="utf-8")
        return pathlib.Path(d)

    _DISCLAIMERS = ("not legal advice not a dense retriever candidate lists only lift over ")

    def test_recommended_requires_passing_gate(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as rd:
            root = self._card(d, self._DISCLAIMERS + "Recommended for German FAQ/RAG reranking now")
            (pathlib.Path(rd) / "eval").mkdir(parents=True)
            (pathlib.Path(rd) / "eval" / "rag_reranker_gate.json").write_text(
                json.dumps({"status": "fail"}), "utf-8")
            issues = VR.check_v4_rag_card(root, pathlib.Path(rd))
            self.assertTrue(any(i[0] == "v4_card_recommended_without_passing_gate" for i in issues))

    def test_recommended_ok_when_gate_passes(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as rd:
            root = self._card(d, self._DISCLAIMERS + "Recommended for German FAQ/RAG reranking now")
            (pathlib.Path(rd) / "eval").mkdir(parents=True)
            (pathlib.Path(rd) / "eval" / "rag_reranker_gate.json").write_text(
                json.dumps({"status": "pass"}), "utf-8")
            self.assertEqual(VR.check_v4_rag_card(root, pathlib.Path(rd)), [])

    def test_missing_disclaimer_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            root = self._card(d, "experimental reranker, not legal advice")   # missing others
            issues = VR.check_v4_rag_card(root, FIX)
            self.assertTrue(any(i[0] == "v4_card_missing_disclaimer" for i in issues))

    def test_shipped_card_experimental_passes(self):
        # the real shipped card is experimental (no literal recommended phrase) + has disclaimers
        self.assertEqual(VR.check_v4_rag_card(VR.ROOT, FIX), [])


class TestGateWiring(unittest.TestCase):
    def test_require_v4_passes_on_fixture(self):
        rep = VR.run_checks(results_dir=FIX, require_v4_rag_artifacts=True)
        self.assertEqual(rep["checks"]["v4_rag_artifacts"], [])
        self.assertEqual(rep["checks"]["v4_rag_card"], [])

    def test_v4_ignores_gerdalir(self):
        # the v4 checks never reference gerdalir/legal — the diagnostic gerdalir report in the
        # fixture does not add any issue.
        rep = VR.run_checks(results_dir=FIX, require_v4_rag_artifacts=True)
        joined = json.dumps(rep["checks"]["v4_rag_artifacts"]) + json.dumps(rep["checks"]["v4_rag_card"])
        self.assertNotIn("gerdalir", joined.lower())

    def test_always_on_gate_does_not_trigger_v4(self):
        rep = VR.run_checks()
        self.assertNotIn("v4_rag_artifacts", rep["checks"])
        self.assertEqual(rep["status"], "pass")   # always-on stays green


if __name__ == "__main__":
    unittest.main()
