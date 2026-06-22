"""Tests for the canonical AutoResearch scorer (stdlib, deterministic, no ML)."""
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load("ar_score")


def _doc(webfaq, gq=0.88, dt=0.95, retention=0.96, leakage=0, vram=10.0, tput=1000.0,
         status="ok", local_rag=None):
    metrics = {
        "webfaq": webfaq,
        "germanquad": {"ndcg@10": gq},
        "dt_test": {"ndcg@10": dt},
        "matryoshka": {"retention_256": retention},
        "leakage": {"hits": leakage},
        "system": {"vram_gb": vram, "throughput_pairs_per_sec": tput},
    }
    if local_rag is not None:
        metrics["local_rag"] = local_rag
    return {"run_id": "t", "status": status, "metrics": metrics}


BASE_WF = {"recall@100": 0.90, "ndcg@10": 0.60, "mrr@10": 0.50}


class ScoreFormulaTests(unittest.TestCase):
    def test_exact_weighted_score(self):
        base = _doc(dict(BASE_WF))
        run = _doc({"recall@100": 0.95, "ndcg@10": 0.66, "mrr@10": 0.55},
                   retention=0.97, vram=12.0, tput=900.0)
        r = S.score_run(run, base)
        # 2*0.05 + 1.5*0.06 + 0.5*0.05 - 0.2*(2/10) - 0.2*(100/1000) = 0.155
        self.assertAlmostEqual(r["score"], 0.155, places=6)
        self.assertEqual(r["status"], "pass")
        self.assertFalse(r["has_local_rag"])

    def test_metric_aliases(self):
        base = _doc(dict(BASE_WF))
        run = _doc({"recall_at_100": 0.95, "ndcg_at_10": 0.66, "mrr_at_10": 0.55})
        run["metrics"]["matryoshka"] = {"retention256": 0.97}
        r = S.score_run(run, base)
        self.assertEqual(r["status"], "pass")
        self.assertAlmostEqual(r["deltas"]["webfaq_recall@100"], 0.05, places=6)
        self.assertIsNotNone(r["deltas"]["webfaq_ndcg@10"])

    def test_local_rag_term_when_present(self):
        base = _doc(dict(BASE_WF), local_rag={"recall@100": 0.50, "ndcg@10": 0.40})
        run = _doc(dict(BASE_WF), local_rag={"recall@100": 0.60, "ndcg@10": 0.45})
        r = S.score_run(run, base)
        self.assertTrue(r["has_local_rag"])
        self.assertAlmostEqual(r["deltas"]["local_rag_recall@100"], 0.10, places=6)
        self.assertAlmostEqual(r["score"], 1.0 * 0.10, places=6)


class GateTests(unittest.TestCase):
    def test_pass_when_equal(self):
        base = _doc(dict(BASE_WF))
        self.assertEqual(S.score_run(base, base)["status"], "pass")

    def test_leakage_failure(self):
        base = _doc(dict(BASE_WF))
        run = _doc(dict(BASE_WF), leakage=2)
        r = S.score_run(run, base)
        self.assertEqual(r["status"], "fail")
        self.assertIn("leakage", [g["name"] for g in r["failed_gates"]])

    def test_germanquad_regression_fails(self):
        base = _doc(dict(BASE_WF))
        run = _doc(dict(BASE_WF), gq=0.87)   # Δ -0.01 < -0.005
        r = S.score_run(run, base)
        self.assertEqual(r["status"], "fail")
        self.assertIn("germanquad_ndcg@10_delta", [g["name"] for g in r["failed_gates"]])

    def test_dt_test_regression_fails(self):
        base = _doc(dict(BASE_WF))
        run = _doc(dict(BASE_WF), dt=0.94)   # Δ -0.01 < -0.005
        self.assertIn("dt_test_ndcg@10_delta",
                      [g["name"] for g in S.score_run(run, base)["failed_gates"]])

    def test_matryoshka_retention_fails(self):
        base = _doc(dict(BASE_WF))
        run = _doc(dict(BASE_WF), retention=0.93)
        self.assertIn("matryoshka_256_retention",
                      [g["name"] for g in S.score_run(run, base)["failed_gates"]])

    def test_missing_webfaq_metrics_fails(self):
        base = _doc(dict(BASE_WF))
        run = _doc({"mrr@10": 0.5})   # no recall@100 / ndcg@10
        names = [g["name"] for g in S.score_run(run, base)["failed_gates"]]
        self.assertIn("webfaq_recall@100_present", names)
        self.assertIn("webfaq_ndcg@10_present", names)

    def test_crash_status_fails(self):
        base = _doc(dict(BASE_WF))
        run = _doc(dict(BASE_WF), status="crash")
        self.assertEqual(S.score_run(run, base)["status"], "fail")


class FailClosedTests(unittest.TestCase):
    """Gates that must FAIL when a safety condition is merely unverified (not just violated)."""

    def test_missing_leakage_block_fails_closed(self):
        base = _doc(dict(BASE_WF))
        run = _doc(dict(BASE_WF))
        del run["metrics"]["leakage"]                       # leakage never measured
        r = S.score_run(run, base)
        self.assertEqual(r["status"], "fail")
        self.assertIn("leakage", [g["name"] for g in r["failed_gates"]])

    def test_not_checked_leakage_status_fails_closed(self):
        base = _doc(dict(BASE_WF))
        run = _doc(dict(BASE_WF))
        run["metrics"]["leakage"] = {"hits": 0, "status": "not_checked"}
        self.assertIn("leakage", [g["name"] for g in S.score_run(run, base)["failed_gates"]])

    def test_verified_clean_leakage_passes(self):
        base = _doc(dict(BASE_WF))
        run = _doc(dict(BASE_WF))
        run["metrics"]["leakage"] = {"hits": 0, "status": "clean"}
        self.assertEqual(S.score_run(run, base)["status"], "pass")

    def test_dry_run_mode_cannot_pass(self):
        base = _doc(dict(BASE_WF))
        run = _doc(dict(BASE_WF))
        run["mode"] = "dry_run"
        r = S.score_run(run, base)
        self.assertEqual(r["status"], "fail")
        self.assertIn("not_a_real_run", [g["name"] for g in r["failed_gates"]])

    def test_scale_disclaimer_cannot_pass(self):
        base = _doc(dict(BASE_WF))
        run = _doc(dict(BASE_WF))
        run["scale_disclaimer"] = "plumbing only"
        self.assertIn("not_a_real_run", [g["name"] for g in S.score_run(run, base)["failed_gates"]])

    def test_absent_baseline_webfaq_fails(self):
        base = _doc(dict(BASE_WF))
        del base["metrics"]["webfaq"]
        run = _doc(dict(BASE_WF))
        r = S.score_run(run, base)
        self.assertIn("baseline_incomplete", [g["name"] for g in r["failed_gates"]])

    def test_zero_skeleton_baseline_fails(self):
        base = _doc({"recall@100": 0.0, "ndcg@10": 0.0, "mrr@10": 0.0})
        run = _doc(dict(BASE_WF))
        r = S.score_run(run, base)
        self.assertIn("baseline_incomplete", [g["name"] for g in r["failed_gates"]])


if __name__ == "__main__":
    unittest.main()
