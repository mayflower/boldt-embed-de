"""Scope-reset (v6) gate tests: the release gate must not steer toward a policy-gated serving
workaround, must keep the raw reranker not-recommended until its raw gate passes, and must let the
dense embedder be recommended independently of the reranker. Stdlib, no ML."""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import validate_release_2026 as V  # noqa: E402


class PolicyGatedServingBanned(unittest.TestCase):
    def test_card_cannot_say_recommended_with_policy(self):
        for text in (
            f"# card\n{V.V5_RECOMMENDED_PHRASE}.\n",
            "# card\nThe reranker is recommended with the bounded policy for production.\n",
            "# card\nWe recommend the policy-gated reranker as the production default.\n",
            "# card\nUse the bounded margin_override policy — recommended for production.\n",
        ):
            issues = V.check_no_policy_gated_recommendation("Boldt-Reranker-DE-350M-v1.md", text)
            self.assertTrue(issues, f"should flag policy-gated recommendation: {text!r}")
            self.assertTrue(all(k == "card_recommends_policy_gated_serving" for k, _ in issues))

    def test_diagnostic_policy_mention_is_allowed(self):
        # Policy work may be MENTIONED as diagnostics/analysis — negated/diagnostic lines are fine.
        for text in (
            "# card\nThe bounded margin_override policy is **diagnostic only** and is **not** a "
            "production recommendation.\n",
            "# card\nPolicy experiments (rerank-or-abstain) are diagnostics; we do not recommend a "
            "policy-gated serving workaround.\n",
            "# card\nThe abstention policy is experimental and never recommended for production.\n",
        ):
            self.assertEqual(
                V.check_no_policy_gated_recommendation("Boldt-Reranker-DE-350M-v1.md", text), [],
                f"diagnostic policy mention must be allowed: {text!r}")

    def test_real_reranker_card_has_no_policy_recommendation(self):
        text = (ROOT / "model_cards" / "Boldt-Reranker-DE-350M-v1.md").read_text(encoding="utf-8")
        self.assertEqual(
            V.check_no_policy_gated_recommendation("Boldt-Reranker-DE-350M-v1.md", text), [])


class RawRerankerRecommendationGated(unittest.TestCase):
    def test_raw_recommendation_blocked_until_raw_gate_passes(self):
        text = f"# card\n{V.V5_RAW_RECOMMENDED_PHRASE} for production.\n"
        # raw gate failing / absent -> not recommended
        self.assertTrue(V.check_reranker_raw_recommendation(
            text, v4_gate_passed=False, v5_raw_gate_passed=False))
        self.assertTrue(V.check_reranker_raw_recommendation(
            text, v4_gate_passed=None, v5_raw_gate_passed=None))
        # raw gate passing -> allowed
        self.assertEqual(V.check_reranker_raw_recommendation(
            text, v4_gate_passed=None, v5_raw_gate_passed=True), [])

    def test_v4_raw_phrase_also_gated(self):
        text = "# card\nRecommended for German FAQ/RAG reranking.\n"
        self.assertTrue(V.check_reranker_raw_recommendation(
            text, v4_gate_passed=False, v5_raw_gate_passed=None))
        self.assertEqual(V.check_reranker_raw_recommendation(
            text, v4_gate_passed=True, v5_raw_gate_passed=None), [])

    def test_real_repo_keeps_raw_reranker_not_recommended(self):
        # The real raw v5 gate did not pass; the card must not carry a raw recommendation.
        self.assertIsNot(V._v5_raw_gate_passed(ROOT), True)
        text = (ROOT / "model_cards" / "Boldt-Reranker-DE-350M-v1.md").read_text(encoding="utf-8")
        self.assertEqual(V.check_reranker_raw_recommendation(
            text, v4_gate_passed=V._v4_gate_passed(ROOT / "outputs" / "v4-rag-reranker"),
            v5_raw_gate_passed=V._v5_raw_gate_passed(ROOT)), [])


class DenseEmbedderIndependentOfReranker(unittest.TestCase):
    def test_embedder_can_be_recommended_while_reranker_is_not(self):
        # A dense-embedder recommendation must NOT be blocked by the reranker's (failing) status.
        embedder_card = ("# Boldt-Embed-DE-350M-v1-causal\nThe dense embedder is **recommended** for "
                         "German RAG first-stage retrieval and is production-ready.\n")
        # the embedder recommendation triggers no policy/raw-reranker block
        self.assertEqual(
            V.check_no_policy_gated_recommendation("Boldt-Embed-DE-350M-v1-causal.md", embedder_card),
            [])
        # the raw-reranker gate only inspects reranker phrases, so an embedder card is unaffected
        self.assertEqual(V.check_reranker_raw_recommendation(
            embedder_card, v4_gate_passed=False, v5_raw_gate_passed=False), [])

    def test_embedder_recommendation_is_not_a_reranker_phrase(self):
        # guard: the embedder "recommended" wording is distinct from any gated reranker phrase
        embedder_card = "# embed\nThe dense embedder is recommended for German retrieval.\n"
        self.assertNotIn("Recommended for German FAQ/RAG reranking", embedder_card)
        self.assertNotIn(V.V5_RAW_RECOMMENDED_PHRASE, embedder_card)
        self.assertNotIn(V.V5_RECOMMENDED_PHRASE, embedder_card)


if __name__ == "__main__":
    unittest.main()
