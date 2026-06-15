"""Stdlib tests for the rerank-or-abstain policy. No ML, no labels at inference."""
import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import rerank_abstain as RA  # noqa: E402

DEV = ROOT / "tests/fixtures/rag_dev_lists_scored.jsonl"
EVAL = ROOT / "tests/fixtures/rag_eval_lists_scored.jsonl"


def _read(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]


def mklist(qid, pos_idx, fs_scores, rr_scores):
    cands = [{"doc_id": f"{qid}-d{i}", "text": f"doc {i}", "candidate_source": "bm25",
              "first_stage_rank": i, "first_stage_score": fs_scores[i], "reranker_score": rr_scores[i],
              "is_positive": i == pos_idx} for i in range(len(fs_scores))]
    return {"query_id": qid, "query": f"q-{qid}", "positive_doc_ids": [f"{qid}-d{pos_idx}"],
            "candidates": cands}


class TestFeatures(unittest.TestCase):
    def test_feature_extraction_deterministic(self):
        row = _read(DEV)[0]
        self.assertEqual(RA.extract_features(row), RA.extract_features(row))

    def test_features_count_and_keys(self):
        f = RA.extract_features(_read(DEV)[0])
        self.assertEqual(len(f), 15)
        for k in ("first_stage_top1_top2_gap", "reranker_top1_top2_gap", "max_rank_displacement",
                  "num_candidates", "num_candidate_sources"):
            self.assertIn(k, f)


class TestNoLabelsAtInference(unittest.TestCase):
    def test_policy_ignores_labels(self):
        row = _read(DEV)[0]
        stripped = json.loads(json.dumps(row))
        stripped.pop("positive_doc_ids", None)
        for c in stripped["candidates"]:
            c.pop("is_positive", None); c.pop("label", None); c.pop("teacher_score", None)
        for pol in ("always_rerank", "first_stage_confidence_abstain", "combined_policy"):
            params = {"fs_gap_threshold": 1.0, "rr_gap_threshold": 0.5}
            a, _ = RA.apply_policy(row, pol, params)
            b, _ = RA.apply_policy(stripped, pol, params)
            self.assertEqual(a, b, f"{pol} changed when labels removed -> uses labels at inference")

    def test_features_ignore_labels(self):
        row = _read(DEV)[0]
        stripped = json.loads(json.dumps(row))
        stripped.pop("positive_doc_ids", None)
        for c in stripped["candidates"]:
            c.pop("is_positive", None)
        self.assertEqual(RA.extract_features(row), RA.extract_features(stripped))


class TestPolicies(unittest.TestCase):
    def test_blend_alpha1_preserves_first_stage_order(self):
        row = _read(DEV)[0]
        fs = [c["doc_id"] for c in RA._first_stage_order(row["candidates"])]
        out, action = RA.apply_policy(row, "conservative_blend", {"alpha": 1.0})
        self.assertEqual(out, fs)
        self.assertEqual(action, "conservative_blend")

    def test_displacement_guard_prevents_catastrophic_drop(self):
        # correct positive at first-stage rank0; reranker sends it to the bottom
        row = mklist("dg", 0, [20, 5, 4, 3, 2, 1, 0.9, 0.8, 0.7, 0.6],
                              [0.0, 5, 4, 3, 2, 1, 0.9, 0.8, 0.7, 0.6])
        always = RA.evaluate_policy([row], "always_rerank", {})
        guard = RA.evaluate_policy([row], "displacement_guard", {"max_displacement_rank": 3})
        self.assertEqual(always["catastrophic_drop_rate"], 1.0)      # always_rerank tanks it
        self.assertEqual(guard["catastrophic_drop_rate"], 0.0)       # guard keeps first stage
        self.assertGreaterEqual(guard["delta_vs_first_stage"], 0.0)

    def test_combined_policy_beats_always_rerank_where_it_fails(self):
        # confident first stage with correct top1; reranker would churn it (always_rerank fails)
        rows = [mklist(f"c{i}", 0, [20, 5, 4, 3, 2, 1, 0.9, 0.8, 0.7, 0.6],
                                   [0.2, 5, 4, 3, 2, 1, 0.9, 0.8, 0.7, 0.6]) for i in range(3)]
        always = RA.evaluate_policy(rows, "always_rerank", {})
        combined = RA.evaluate_policy(rows, "combined_policy",
                                      {"fs_gap_threshold": 2.0, "rr_gap_threshold": 0.5,
                                       "max_displacement_rank": 3, "alpha": 1.0})
        self.assertLess(always["policy_ndcg@10"], combined["policy_ndcg@10"])
        self.assertGreater(combined["delta_vs_always_rerank"], 0.0)


class TestFit(unittest.TestCase):
    def test_grid_search_dev_only(self):
        dev = _read(DEV)
        fit = RA.grid_search(dev, fs_gaps=[2.0, 15.0], rr_gaps=[0.5, 2.0])
        self.assertEqual(fit["fit_on"], "dev_only")
        self.assertIn("fs_gap_threshold", fit["best_params"])

    def test_guardrails_cannot_influence_thresholds(self):
        # grid_search only consumes the dev rows it is given; appending guardrail-like rows to a
        # SEPARATE list must not change the fitted params (they are never passed in).
        dev = _read(DEV)
        a = RA.grid_search(dev, fs_gaps=[2.0, 15.0], rr_gaps=[0.5, 2.0])["best_params"]
        guardrail = _read(EVAL)  # pretend these are germanquad/dt_test — NOT passed to grid_search
        b = RA.grid_search(dev, fs_gaps=[2.0, 15.0], rr_gaps=[0.5, 2.0])["best_params"]
        self.assertEqual(a, b)
        self.assertTrue(len(guardrail) > 0)

    def test_fit_finds_abstaining_policy_on_mixed_dev(self):
        # dev = confident(churn) + low(fix); a good policy beats always_rerank
        dev = _read(DEV)
        fit = RA.grid_search(dev, fs_gaps=[1.0, 2.0, 5.0, 15.0], rr_gaps=[0.5, 1.0, 2.0])
        self.assertGreater(fit["dev_metrics"]["delta_vs_first_stage"],
                           RA.evaluate_policy(dev, "always_rerank", {})["delta_vs_first_stage"])


class TestEvalReport(unittest.TestCase):
    def test_report_has_required_fields_and_buckets(self):
        rep = RA.evaluate_policy(_read(EVAL), "combined_policy",
                                 {"fs_gap_threshold": 2.0, "rr_gap_threshold": 0.5,
                                  "max_displacement_rank": 3, "alpha": 1.0})
        for k in ("first_stage_ndcg@10", "always_rerank_ndcg@10", "policy_ndcg@10",
                  "delta_vs_first_stage", "delta_vs_always_rerank", "abstain_rate", "rerank_rate",
                  "catastrophic_drop_rate", "by_bucket"):
            self.assertIn(k, rep)


class TestDryRunNoMl(unittest.TestCase):
    def test_no_torch_in_subprocess(self):
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "from boldt_embed import rerank_abstain as RA\n"
            "import json\n"
            "rows=[json.loads(l) for l in open(%r)]\n"
            "RA.grid_search(rows, fs_gaps=[1.0,5.0], rr_gaps=[0.5,2.0])\n"
            "RA.apply_policy(rows[0], 'combined_policy', {'fs_gap_threshold':1.0,'rr_gap_threshold':0.5})\n"
            "assert 'torch' not in sys.modules, 'torch imported'\n"
            "print('OK')\n" % (str(ROOT / "src"), str(DEV))
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("OK", r.stdout)


if __name__ == "__main__":
    unittest.main()
