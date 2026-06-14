"""Tests for v4 RAG reranker training: listwise-primary, BCE restricted to confident labels."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import reranker_modern as RM  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
SCRIPT = ROOT / "scripts" / "train_rag_reranker_v4.py"


def _rows():
    return [json.loads(l) for l in (FIX / "rag_train_scored_tiny.jsonl").read_text("utf-8").splitlines()]


class TestLossPlan(unittest.TestCase):
    def test_mixed_listwise_plan(self):
        p = RM.plan_rag_reranker_loss("mixed_listwise")
        self.assertEqual(p["primary"], "listwise")
        self.assertTrue(p["pointwise_bce_high_confidence_only"])
        self.assertIn("KLDivLoss(listwise)", p["components"])
        # listwise dominates; BCE weight is small
        self.assertEqual(p["weights"]["listwise"], 1.0)
        self.assertLess(p["weights"]["pointwise_bce"], p["weights"]["listwise"])

    def test_with_mse(self):
        p = RM.plan_rag_reranker_loss("mixed_listwise", with_mse=True)
        self.assertTrue(any("MSELoss" in c for c in p["components"]))
        self.assertGreater(p["weights"]["mse"], 0.0)


class TestBuilders(unittest.TestCase):
    def test_listwise_target_sums_to_one(self):
        for b in RM.scored_lists_to_listwise(_rows()):
            self.assertAlmostEqual(sum(b["target"]), 1.0, places=5)
            self.assertGreaterEqual(len(b["documents"]), 2)

    def test_uncertain_excluded_from_bce(self):
        pw = RM.scored_lists_to_pointwise_high_confidence(_rows())
        docs = {e["document"] for e in pw}
        # uncertain candidates (teacher-only-positive + too-close) never enter BCE
        self.assertNotIn("teacher mag das", docs)        # d3 teacher-only positive
        self.assertNotIn("grenzfall", docs)              # d4 too-close
        self.assertNotIn("nah dran", docs)               # d7 too-close
        # a GOLD positive the teacher scored low (d8, not high-precision) is also excluded
        self.assertNotIn("gold aber teacher mag nicht", docs)
        # only high-precision gold (label 1.0) + clear hard negatives (0.0)
        self.assertEqual({e["label"] for e in pw}, {1.0, 0.0})
        self.assertIn("gold antwort", docs)              # d1 high-precision gold
        self.assertEqual(sum(1 for e in pw if e["label"] == 1.0), 2)   # d1, d5

    def test_training_report_counts(self):
        rep = RM.rag_reranker_training_report(_rows())
        self.assertEqual(rep["gold_positives"], 3)
        self.assertEqual(rep["hard_negatives"], 3)
        self.assertEqual(rep["uncertain"], 3)
        self.assertEqual(rep["teacher_only_positives"], 1)   # only d3 (ts 5.0 >= 4)
        self.assertEqual(rep["examples_by_domain"], {"faq_real": 2, "web": 1})
        self.assertGreater(rep["teacher_score_separation"]["separation"], 0)


class TestSampler(unittest.TestCase):
    def test_domain_balanced(self):
        sampled = RM.domain_balanced_list_sampler(_rows(), max_per_domain=1, seed=0)
        doms = [r["domain"] for r in sampled]
        self.assertEqual(sorted(doms), ["faq_real", "web"])   # 1 per domain
        # deterministic
        self.assertEqual([r["query_id"] for r in sampled],
                         [r["query_id"] for r in RM.domain_balanced_list_sampler(_rows(), max_per_domain=1, seed=0)])

    def test_no_cap_keeps_all(self):
        self.assertEqual(len(RM.domain_balanced_list_sampler(_rows())), len(_rows()))


class TestCli(unittest.TestCase):
    def test_listed_dry_run(self):
        out = subprocess.run(
            [sys.executable, str(SCRIPT), "--candidate-lists", str(FIX / "rag_train_scored_tiny.jsonl"),
             "--output", "/tmp/_rag_v4", "--dry-run"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("dry-run-ok", out.stdout)

    def test_eval_leakage_guard_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            ev = pathlib.Path(d) / "eval_ids.txt"
            ev.write_text("q1\n", encoding="utf-8")     # q1 is in the training fixture
            out = subprocess.run(
                [sys.executable, str(SCRIPT), "--candidate-lists", str(FIX / "rag_train_scored_tiny.jsonl"),
                 "--output", "/tmp/_rag_v4", "--eval-query-ids", str(ev), "--dry-run"],
                capture_output=True, text=True)
            self.assertEqual(out.returncode, 2)
            self.assertIn("eval query_id", out.stderr)

    def test_module_is_stdlib_only(self):
        code = ("import sys; sys.path.insert(0, %r);"
                "from boldt_embed import reranker_modern;"
                "assert 'torch' not in sys.modules; print('clean')") % str(ROOT / "src")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


if __name__ == "__main__":
    unittest.main()
