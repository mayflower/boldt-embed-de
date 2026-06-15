"""Stdlib tests for the v5 small-RAG release checks in validate_release_2026. No ML."""
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import validate_release_2026 as V  # noqa: E402


def _root(card_text, gate_status=None, with_artifacts=False):
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "model_cards").mkdir()
    (d / "model_cards" / "Boldt-Reranker-DE-350M-v1.md").write_text(card_text, encoding="utf-8")
    (d / "configs" / "experiments").mkdir(parents=True)
    if with_artifacts:
        (d / "configs" / "experiments" / "v5_small_rag.json").write_text("{}", encoding="utf-8")
        v5 = d / "outputs" / "v5-small-rag"
        for a in V.V5_SMALL_RAG_REQUIRED_ARTIFACTS:
            p = v5 / a
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}", encoding="utf-8")
    if gate_status is not None:
        g = d / "outputs" / "v5-small-rag" / "abstain" / "gate.json"
        g.parent.mkdir(parents=True, exist_ok=True)
        g.write_text(json.dumps({"status": gate_status}), encoding="utf-8")
    return d


class TestV5Card(unittest.TestCase):
    def test_recommended_without_passing_gate_is_flagged(self):
        card = f"# card\n{V.V5_RECOMMENDED_PHRASE}.\n"
        issues = V.check_v5_small_rag_card(_root(card, gate_status="fail"))
        self.assertTrue(any(k == "v5_card_recommended_without_passing_gate" for k, _ in issues))

    def test_recommended_with_passing_gate_is_allowed(self):
        card = f"# card\n{V.V5_RECOMMENDED_PHRASE}.\n"
        issues = V.check_v5_small_rag_card(_root(card, gate_status="pass"))
        self.assertFalse(any(k == "v5_card_recommended_without_passing_gate" for k, _ in issues))

    def test_experimental_card_is_clean(self):
        card = "# card\nExperimental; not recommended for production reranking.\n"
        self.assertEqual(V.check_v5_small_rag_card(_root(card, gate_status="fail")), [])

    def test_never_recommend_always_rerank(self):
        card = "# card\nWe recommend always-rerank for everything.\n"
        issues = V.check_v5_small_rag_card(_root(card, gate_status="pass"))
        self.assertTrue(any(k == "v5_card_recommends_always_rerank" for k, _ in issues))


class TestV5Artifacts(unittest.TestCase):
    def test_missing_artifacts_flagged(self):
        issues = V.check_v5_small_rag_artifacts(_root("# card\n", with_artifacts=False))
        self.assertTrue(issues)

    def test_present_artifacts_clean(self):
        issues = V.check_v5_small_rag_artifacts(_root("# card\n", with_artifacts=True))
        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
