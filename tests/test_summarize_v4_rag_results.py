"""Tests for the v4 RAG results summary. Pure stdlib."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import summarize_v4_rag_results as S  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "v4_rag_results"
SCRIPT = ROOT / "scripts" / "summarize_v4_rag_results.py"


def _lift(es, delta, recall=0.9, diag=False):
    return {"eval_set": es, "diagnostic": diag, "fixed_candidates": True,
            "first_stage_ndcg@10": 0.8, "reranked_ndcg@10": round(0.8 + delta, 4),
            "delta_ndcg@10": delta, "first_stage_mrr@10": 0.78, "reranked_mrr@10": 0.8,
            "positive_in_top_10_before": recall, "positive_in_top_10_after": recall,
            "answer_support_at_10": recall, "oracle_ndcg@10": 0.99, "first_stage_recall_top_10": recall}


def _mk_workdir(d, lifts, gate=None):
    base = pathlib.Path(d)
    (base / "eval").mkdir(parents=True, exist_ok=True)
    for r in lifts:
        (base / "eval" / f"reranker_lift_{r['eval_set']}.json").write_text(json.dumps(r), "utf-8")
    if gate is not None:
        (base / "eval" / "rag_reranker_gate.json").write_text(json.dumps(gate), "utf-8")
    return base


class TestVerdict(unittest.TestCase):
    def test_promoted_on_fixture(self):
        s = S.summarize(FIX)
        self.assertEqual(s["verdict"], "promoted")
        self.assertEqual(s["gate_status"], "pass")
        self.assertIn("gerdalir", s["diagnostic_sets"])

    def test_mixed_when_rag_lifts_but_benchmark_degrades(self):
        with tempfile.TemporaryDirectory() as d:
            # no gate file -> summary recomputes via evaluate_promotion; gq degrades -> gate fail
            _mk_workdir(d, [_lift("webfaq", 0.05), _lift("germanquad", -0.01, 0.95),
                            _lift("dt_test", 0.0, 0.95)])
            s = S.summarize(pathlib.Path(d))
            self.assertEqual(s["verdict"], "mixed")        # RAG lift present but gate fails
            self.assertIn("germanquad", s["failure_cases"]["reranker_hurts"])

    def test_not_promoted_when_no_rag_lift(self):
        with tempfile.TemporaryDirectory() as d:
            _mk_workdir(d, [_lift("webfaq", 0.0), _lift("germanquad", 0.0, 0.95),
                            _lift("dt_test", 0.0, 0.95)])
            self.assertEqual(S.summarize(pathlib.Path(d))["verdict"], "not_promoted")


class TestGraceful(unittest.TestCase):
    def test_missing_local_rag_handled(self):
        s = S.summarize(FIX)                                # fixture has no local_rag
        self.assertNotIn("local_rag", s["reranker_lift"])
        self.assertNotIn("local_rag", s["training_data"]["excluded_eval_splits"])
        # still produces a full report
        self.assertIn("webfaq", s["reranker_lift"])

    def test_empty_workdir_no_crash(self):
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "eval").mkdir()
            s = S.summarize(pathlib.Path(d))
            self.assertEqual(s["verdict"], "not_promoted")
            self.assertEqual(s["gate_status"], "missing")


class TestMarkdown(unittest.TestCase):
    def test_tables_stable_and_sectioned(self):
        s = S.summarize(FIX)
        md1 = S.render_markdown(s)
        md2 = S.render_markdown(s)
        self.assertEqual(md1, md2)                          # deterministic
        for header in ("## 1. Executive verdict", "## 2. Training data", "## 3. Reranker lift",
                       "## 4. First-stage recall", "## 5. Teacher / student",
                       "## 6. Failure cases", "## 7. Decision"):
            self.assertIn(header, md1)
        self.assertIn("| eval set | first stage | reranked | delta | diagnostic |", md1)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            md, js = pathlib.Path(d) / "r.md", pathlib.Path(d) / "r.json"
            out = subprocess.run([sys.executable, str(SCRIPT), "--work-dir", str(FIX),
                                  "--output", str(md), "--json-output", str(js)],
                                 capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(json.loads(js.read_text("utf-8"))["verdict"], "promoted")


if __name__ == "__main__":
    unittest.main()
