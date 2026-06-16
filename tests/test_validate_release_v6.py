"""Stdlib tests for the v6 active-track release checks in validate_release_2026 (dense embedder +
RAW reranker). No ML."""
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import validate_release_2026 as V  # noqa: E402


def _root(*, embed_card="# embed\n", reranker_card="# reranker\nnot recommended.\n",
          dense_gate=None, raw_gate=None, recall_report=True, lift_modes=None):
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "model_cards").mkdir()
    (d / "model_cards" / "Boldt-Embed-DE-350M-v1-causal.md").write_text(embed_card, encoding="utf-8")
    (d / "model_cards" / "Boldt-Reranker-DE-350M-v1.md").write_text(reranker_card, encoding="utf-8")
    dd = d / "outputs" / "v6-dense-rag"; dd.mkdir(parents=True)
    rd = d / "outputs" / "v6-reranker" / "eval"; rd.mkdir(parents=True)
    if recall_report:
        (dd / "webfaq_real_recall_bm25_vs_dense.json").write_text(json.dumps(
            {"dense_v6": {"recall@100": 0.96, "recall@50": 0.88}}), encoding="utf-8")
    (dd / "first_stage_audit_webfaq.json").write_text("{}", encoding="utf-8")
    if dense_gate is not None:
        (dd / "dense_recall_gate.json").write_text(json.dumps({"status": dense_gate}), encoding="utf-8")
    if raw_gate is not None:
        (rd.parent / "raw_gate.json").write_text(json.dumps(
            {"status": raw_gate, "evaluated_ranking_mode": "raw", "policy_gated_result_used": False,
             "checks": [{"check": "no_public_eval_leakage", "status": "pass", "detail": ""}]}),
            encoding="utf-8")
    for s, mode in (lift_modes or {}).items():
        (rd / f"{s}_lift.json").write_text(json.dumps({"eval_set": s, "ranking_mode": mode}),
                                           encoding="utf-8")
    return d


class DenseRecommendationTests(unittest.TestCase):
    def test_recommended_without_passing_gate_flagged(self):
        card = f"# embed\n{V.V6_DENSE_RECOMMENDED_PHRASE} for production.\n"
        issues = V.check_v6_dense_recommendation(_root(embed_card=card, dense_gate="fail"))
        self.assertTrue(any(k == "dense_recommended_without_passing_dense_gate" for k, _ in issues))

    def test_recommended_with_passing_gate_and_reports_ok(self):
        card = f"# embed\n{V.V6_DENSE_RECOMMENDED_PHRASE} for production.\n"
        self.assertEqual(
            V.check_v6_dense_recommendation(_root(embed_card=card, dense_gate="pass")), [])

    def test_recommended_without_reports_flagged(self):
        card = f"# embed\n{V.V6_DENSE_RECOMMENDED_PHRASE}.\n"
        issues = V.check_v6_dense_recommendation(
            _root(embed_card=card, dense_gate="pass", recall_report=False))
        self.assertTrue(any(k == "dense_recommended_without_recall_reports" for k, _ in issues))

    def test_no_claim_is_clean(self):
        self.assertEqual(V.check_v6_dense_recommendation(_root(dense_gate="fail")), [])


class RerankerRecommendationTests(unittest.TestCase):
    def test_recommended_without_passing_raw_gate_flagged(self):
        card = f"# rr\n{V.V6_RAW_RECOMMENDED_PHRASE}.\n"
        issues = V.check_v6_raw_reranker_recommendation(_root(reranker_card=card, raw_gate="fail"))
        self.assertTrue(any(k == "v6_reranker_recommended_without_passing_raw_gate" for k, _ in issues))

    def test_recommended_with_passing_raw_gate_ok(self):
        card = f"# rr\n{V.V6_RAW_RECOMMENDED_PHRASE}.\n"
        self.assertEqual(
            V.check_v6_raw_reranker_recommendation(_root(reranker_card=card, raw_gate="pass")), [])


class PolicyEvidenceTests(unittest.TestCase):
    def test_policy_mode_lift_flagged(self):
        r = _root(lift_modes={"webfaq": "bounded_margin_override", "germanquad": "raw"})
        issues = V.check_no_policy_result_as_promotion_evidence(r)
        self.assertTrue(any(k == "policy_result_as_promotion_evidence" for k, _ in issues))

    def test_raw_mode_lift_ok(self):
        r = _root(lift_modes={"webfaq": "raw", "germanquad": "raw"}, raw_gate="fail")
        self.assertEqual(V.check_no_policy_result_as_promotion_evidence(r), [])


class LeakageTests(unittest.TestCase):
    def test_public_eval_leakage_flagged(self):
        d = _root(raw_gate="fail")
        gate = d / "outputs" / "v6-reranker" / "raw_gate.json"
        g = json.loads(gate.read_text())
        g["checks"] = [{"check": "no_public_eval_leakage", "status": "fail", "detail": "leak!"}]
        gate.write_text(json.dumps(g))
        issues = V.check_no_public_eval_leakage_v6(d)
        self.assertTrue(any(k == "public_eval_leakage" for k, _ in issues))


class ArtifactTests(unittest.TestCase):
    def test_dense_artifacts_missing_and_present(self):
        empty = pathlib.Path(tempfile.mkdtemp())
        self.assertTrue(V.check_v6_dense_artifacts(empty))
        d = _root(dense_gate="fail")
        self.assertEqual(V.check_v6_dense_artifacts(d), [])      # gate + recall + audit present

    def test_raw_reranker_artifacts_missing_and_present(self):
        empty = pathlib.Path(tempfile.mkdtemp())
        self.assertTrue(V.check_v6_raw_reranker_artifacts(empty))
        d = _root(raw_gate="fail", lift_modes={"webfaq": "raw", "germanquad": "raw", "dt_test": "raw"})
        self.assertEqual(V.check_v6_raw_reranker_artifacts(d), [])


class RealRepoTests(unittest.TestCase):
    def test_real_cards_have_no_ungated_recommendation(self):
        # the real embedder/reranker cards must not claim a recommendation while gates fail
        self.assertEqual(V.check_v6_dense_recommendation(ROOT), [])
        self.assertEqual(V.check_v6_raw_reranker_recommendation(ROOT), [])
        self.assertEqual(V.check_no_policy_result_as_promotion_evidence(ROOT), [])
        self.assertEqual(V.check_no_public_eval_leakage_v6(ROOT), [])

    def test_real_gates_currently_fail(self):
        # documents the honest state: both v6 gates are not 'pass' right now
        self.assertIsNot(V._v6_dense_gate_passed(ROOT), True)
        self.assertIsNot(V._v6_raw_reranker_gate_passed(ROOT), True)


if __name__ == "__main__":
    unittest.main()
