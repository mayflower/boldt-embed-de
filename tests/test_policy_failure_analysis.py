"""Tests for the v5 frozen-policy failure analyzer (stdlib, no ML)."""
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, sub="scripts"):
    spec = importlib.util.spec_from_file_location(name, ROOT / sub / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AN = _load("analyze_policy_gate_failures")


def _facts(**kw):
    base = {
        "query_id": "q", "domain": "faq", "fs_ndcg": 0.5, "raw_ndcg": 0.5, "pol_ndcg": 0.5,
        "fs_rank_pos": 1, "rr_rank_pos": 1, "pol_rank_pos": 1, "preserve_k": 3, "margin": 3.0,
        "max_up": 5, "margin_override_used": False, "top_k_locked": 3, "override_is_positive": False,
        "override_doc_nonpos": False, "best_pos_rr_gap": 0.0, "displacer_source": None,
        "best_pos_action": "locked", "degenerate_fs": False, "near_dup_top": False,
        "num_candidates": 20,
    }
    base.update(kw)
    return base


class ClassifyTests(unittest.TestCase):
    def test_not_a_failure_returns_none(self):
        self.assertIsNone(AN.classify_failure(_facts(fs_ndcg=0.5, raw_ndcg=0.5, pol_ndcg=0.5)))

    def test_no_useful_first_stage_score(self):
        # missed lift + degenerate first-stage order -> calibration, not policy-fixable
        c = AN.classify_failure(_facts(fs_ndcg=0.0, raw_ndcg=1.0, pol_ndcg=0.0, degenerate_fs=True))
        self.assertEqual(c[1], "no_useful_first_stage_score")
        self.assertFalse(c[3])
        self.assertEqual(AN.CAT_RECO[c[1]], "add_calibration_features")

    def test_margin_override_too_permissive_is_a_regression(self):
        # mild regression (not catastrophic) so ftype is exactly "regression"
        c = AN.classify_failure(_facts(fs_ndcg=1.0, raw_ndcg=0.95, pol_ndcg=0.9,
                                       margin_override_used=True, override_doc_nonpos=True))
        self.assertEqual(c[1], "margin_override_too_permissive")
        self.assertEqual(c[0], "regression")
        self.assertTrue(c[3])

    def test_top_k_lock_too_strict(self):
        # positive in the tail (rank 5), raw lifts to 0, head locked at top-3
        c = AN.classify_failure(_facts(fs_ndcg=0.3, raw_ndcg=1.0, pol_ndcg=0.3,
                                       fs_rank_pos=5, rr_rank_pos=0, pol_rank_pos=5))
        self.assertEqual(c[1], "top_k_lock_too_strict")
        self.assertTrue(c[3])

    def test_margin_override_too_strict(self):
        # reranker gap 2.0 < margin 3.0; positive identified but override didn't fire
        c = AN.classify_failure(_facts(fs_ndcg=0.3, raw_ndcg=0.9, pol_ndcg=0.3, fs_rank_pos=4,
                                       rr_rank_pos=0, pol_rank_pos=4, best_pos_rr_gap=2.0,
                                       margin_override_used=False))
        self.assertEqual(c[1], "margin_override_too_strict")
        self.assertTrue(c[3])

    def test_positive_locked_too_low(self):
        c = AN.classify_failure(_facts(fs_ndcg=0.63, raw_ndcg=1.0, pol_ndcg=0.65, fs_rank_pos=2,
                                       rr_rank_pos=0, pol_rank_pos=2, best_pos_action="locked"))
        self.assertEqual(c[1], "positive_locked_too_low")

    def test_duplicate_near_duplicate_confusion(self):
        c = AN.classify_failure(_facts(fs_ndcg=0.4, raw_ndcg=0.9, pol_ndcg=0.4, near_dup_top=True))
        self.assertEqual(c[1], "duplicate_near_duplicate_confusion")
        self.assertEqual(AN.CAT_RECO[c[1]], "add_more_data")

    def test_candidate_source_artifact(self):
        # not explained by lock/margin/blend, but a specific source displaced the positive
        c = AN.classify_failure(_facts(fs_ndcg=0.5, raw_ndcg=0.9, pol_ndcg=0.5, fs_rank_pos=0,
                                       rr_rank_pos=0, pol_rank_pos=2, best_pos_action=None,
                                       displacer_source="manual"))
        self.assertEqual(c[1], "candidate_source_artifact")
        self.assertEqual(AN.CAT_RECO[c[1]], "add_more_data")

    def test_unknown_fallback(self):
        c = AN.classify_failure(_facts(fs_ndcg=0.5, raw_ndcg=0.9, pol_ndcg=0.5, fs_rank_pos=0,
                                       rr_rank_pos=0, pol_rank_pos=0, best_pos_action=None,
                                       displacer_source=None))
        self.assertEqual(c[1], "unknown")


class AnalyzeSetTests(unittest.TestCase):
    def _row(self, qid, pos_first_stage_rank, *, pos_rr, others_rr, n=20, pos_src="bm25"):
        cands = []
        for i in range(n):
            is_pos = (i == pos_first_stage_rank)
            cands.append({"doc_id": f"{qid}-d{i}", "text": f"t{i}", "first_stage_rank": i,
                          "first_stage_score": float(n - i),
                          "reranker_score": pos_rr if is_pos else others_rr[i % len(others_rr)],
                          "candidate_source": pos_src if is_pos else "bm25"})
        return {"query_id": qid, "domain": "faq", "positive_doc_ids": [f"{qid}-d{pos_first_stage_rank}"],
                "candidates": cands}

    def test_aggregates_by_category_and_source(self):
        policy = AN.load_policy(str(ROOT / "configs/policies/bounded_margin_override_v1.json"))
        # positives deep in the tail with a sub-margin reranker gap -> tunable missed-lift failures
        rows = [self._row(f"q{i}", 7, pos_rr=2.0, others_rr=[0.0]) for i in range(5)]
        rep = AN.analyze_set(rows, policy, "webfaq", max_examples=2)
        self.assertEqual(rep["eval_set"], "webfaq")
        self.assertEqual(rep["n_queries"], 5)
        self.assertGreaterEqual(rep["n_failures"], 1)
        self.assertTrue(rep["by_category"])
        self.assertLessEqual(len(rep["examples"]), 2)
        self.assertIn("by_displacer_source", rep)

    def test_clean_set_has_no_failures(self):
        policy = AN.load_policy(str(ROOT / "configs/policies/bounded_margin_override_v1.json"))
        # positive already at first-stage rank 0 with the top reranker score -> nothing to fix
        rows = [self._row(f"q{i}", 0, pos_rr=10.0, others_rr=[0.0]) for i in range(4)]
        rep = AN.analyze_set(rows, policy, "near_ceiling")
        self.assertEqual(rep["n_failures"], 0)
        self.assertEqual(rep["policy_fixable_share"], 0.0)


if __name__ == "__main__":
    unittest.main()
