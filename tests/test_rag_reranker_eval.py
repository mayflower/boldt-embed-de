"""Tests for v4 RAG reranker lift eval + promotion gate. Pure stdlib."""
import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import rag_reranker_eval as RE  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
LIFT = ROOT / "scripts" / "eval_rag_reranker_lift.py"
GATE = ROOT / "scripts" / "check_rag_reranker_promotion_gate.py"


def _pass_reports():
    return [json.loads((FIX / "rag_lift_pass" / f"reranker_lift_{s}.json").read_text("utf-8"))
            for s in ("webfaq", "germanquad", "dt_test", "gerdalir")]


class TestLift(unittest.TestCase):
    def test_lift_over_fixed_candidates(self):
        rows = [json.loads(l) for l in (FIX / "rag_eval_lists_tiny.jsonl").read_text("utf-8").splitlines()]
        # rerank by the candidates' teacher_score (the dry-run fallback)
        rep = RE.build_lift_report(rows, "webfaq")
        self.assertTrue(rep["fixed_candidates"])
        self.assertGreater(rep["delta_ndcg@10"], 0.0)            # teacher order beats first stage
        self.assertEqual(rep["reranked_ndcg@10"], 1.0)
        self.assertLessEqual(rep["first_stage_ndcg@10"], rep["reranked_ndcg@10"])
        self.assertEqual(rep["positive_in_top_10_after"], 1.0)
        self.assertEqual(rep["oracle_ndcg@10"], 1.0)
        self.assertIsNotNone(rep["answer_support_at_10"])

    def test_no_candidates_not_fixed(self):
        rows = [{"query_id": "q", "query": "x", "positive_doc_ids": ["d1"], "candidates": []}]
        rep = RE.build_lift_report(rows, "webfaq")
        self.assertFalse(rep["fixed_candidates"])
        self.assertEqual(rep["n_queries_without_candidates"], 1)


class TestGate(unittest.TestCase):
    def test_passes_on_lift_and_no_degradation(self):
        self.assertEqual(RE.evaluate_promotion(_pass_reports())["status"], "pass")

    def test_fails_on_germanquad_negative(self):
        reps = _pass_reports()
        for r in reps:
            if r["eval_set"] == "germanquad":
                r["delta_ndcg@10"] = -0.01
        res = RE.evaluate_promotion(reps)
        self.assertEqual(res["status"], "fail")
        self.assertTrue(any(c["check"] == "germanquad_neutral_or_better" for c in res["failing"]))

    def test_fails_on_low_webfaq_lift(self):
        reps = _pass_reports()
        for r in reps:
            if r["eval_set"] == "webfaq":
                r["delta_ndcg@10"] = 0.01           # below 0.03
        self.assertEqual(RE.evaluate_promotion(reps)["status"], "fail")

    def test_ignores_gerdalir_diagnostic(self):
        # gerdalir delta is -0.10 (catastrophic) but diagnostic -> must NOT fail the gate
        res = RE.evaluate_promotion(_pass_reports())
        self.assertEqual(res["status"], "pass")
        self.assertIn("gerdalir", res["diagnostic_sets"])
        self.assertFalse(any("gerdalir" in c["check"] for c in res["failing"]))

    def test_fails_when_not_fixed_candidates(self):
        reps = _pass_reports()
        for r in reps:
            if r["eval_set"] == "webfaq":
                r["fixed_candidates"] = False
        res = RE.evaluate_promotion(reps)
        self.assertEqual(res["status"], "fail")
        self.assertTrue(any(c["check"] == "webfaq_fixed_candidates" for c in res["failing"]))

    def test_fails_on_low_first_stage_recall(self):
        reps = _pass_reports()
        for r in reps:
            if r["eval_set"] == "webfaq":
                r["first_stage_recall_top_10"] = 0.1   # reranking can't matter
        self.assertEqual(RE.evaluate_promotion(reps)["status"], "fail")


class TestCli(unittest.TestCase):
    def test_lift_dry_run_no_ml(self):
        out = subprocess.run(
            [sys.executable, str(LIFT), "--reranker", "/nonexistent",
             "--candidate-lists", str(FIX / "rag_eval_lists_tiny.jsonl"),
             "--output", "/tmp/_rag_lift.json", "--dry-run"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("dry-run-ok", out.stdout)

    def test_gate_cli_pass(self):
        out = subprocess.run(
            [sys.executable, str(GATE), "--eval-dir", str(FIX / "rag_lift_pass"),
             "--output", "/tmp/_rag_gate.json"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("PASS", out.stdout)

    def test_module_is_stdlib_only(self):
        code = ("import sys; sys.path.insert(0, %r);"
                "from boldt_embed import rag_reranker_eval;"
                "assert 'torch' not in sys.modules; print('clean')") % str(ROOT / "src")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


if __name__ == "__main__":
    unittest.main()
