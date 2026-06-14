"""Tests for RAG teacher scoring: listwise targets + high-precision label policy. Pure stdlib."""
import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import rag_teacher_scoring as RT  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
SCRIPT = ROOT / "scripts" / "score_rag_candidate_lists.py"


def _rows():
    return [json.loads(l) for l in (FIX / "rag_lists_tiny.jsonl").read_text("utf-8").splitlines()]


def _by_id(row):
    return {c["doc_id"]: c for c in row["candidates"]}


class TestListwise(unittest.TestCase):
    def test_softmax_target_sums_to_one(self):
        for r in RT.annotate_lists(_rows()):
            s = sum(c["teacher_softmax_target"] for c in r["candidates"])
            self.assertAlmostEqual(s, 1.0, places=5)

    def test_teacher_rank_assigned(self):
        r = RT.annotate_list(_rows()[0])
        ranks = sorted(c["teacher_rank"] for c in r["candidates"])
        self.assertEqual(ranks, [1, 2, 3, 4])
        # highest teacher score gets rank 1 (d1 @ 6.0)
        self.assertEqual(_by_id(r)["d1"]["teacher_rank"], 1)


class TestLabelPolicy(unittest.TestCase):
    def test_gold_positive_and_high_precision(self):
        ann = RT.annotate_lists(_rows())
        d1 = _by_id(ann[0])["d1"]; d8 = _by_id(ann[2])["d8"]
        self.assertEqual(d1["label"], 1)
        self.assertTrue(d1["high_precision_positive"])         # gold @ 6.0 >= 4
        self.assertEqual(d8["label"], 1)                       # gold stays positive
        self.assertFalse(d8["high_precision_positive"])        # but teacher score 1.5 < 4

    def test_teacher_only_positive_is_uncertain_not_negative(self):
        d3 = _by_id(RT.annotate_list(_rows()[0]))["d3"]        # non-gold @ 5.0
        self.assertTrue(d3["uncertain"])
        self.assertIsNone(d3["label"])                         # NOT a hard negative; null
        # with the override it becomes a (listwise) positive
        d3b = _by_id(RT.annotate_list(_rows()[0], use_teacher_only_positives=True))["d3"]
        self.assertEqual(d3b["label"], 1)

    def test_too_close_is_uncertain(self):
        d4 = _by_id(RT.annotate_list(_rows()[0]))["d4"]        # non-gold @ 3.0 (2<3<4)
        self.assertTrue(d4["uncertain"])
        self.assertIsNone(d4["label"])

    def test_strong_negative_margin(self):
        ann = RT.annotate_list(_rows()[0])
        d2 = _by_id(ann)["d2"]                                  # non-gold @ 1.0 (<= 4-2)
        self.assertEqual(d2["label"], 0)
        self.assertFalse(d2["uncertain"])


class TestSummary(unittest.TestCase):
    def test_summary_counts_and_disagreements(self):
        s = RT.summarize(RT.annotate_lists(_rows()))
        self.assertEqual(s["positives"], 3)
        self.assertEqual(s["negatives"], 3)
        self.assertEqual(s["uncertain"], 3)
        self.assertAlmostEqual(s["uncertain_fraction"], 0.3333, places=3)
        kinds = {d["kind"] for d in s["teacher_disagreements"]}
        self.assertEqual(kinds, {"teacher_only_positive", "gold_low_teacher"})
        self.assertIn("faq_real", s["separation_by_domain"])
        self.assertIn("bm25", s["candidate_source_quality"])


class TestCli(unittest.TestCase):
    def test_listed_dry_run(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            out = subprocess.run(
                [sys.executable, str(SCRIPT), "--input", str(FIX / "rag_lists_tiny.jsonl"),
                 "--output", str(pathlib.Path(d) / "s.jsonl"),
                 "--summary", str(pathlib.Path(d) / "sum.json"), "--dry-run"],
                capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("dry-run-ok", out.stdout)
            self.assertEqual(json.loads((pathlib.Path(d) / "sum.json").read_text())["positives"], 3)

    def test_module_is_stdlib_only(self):
        code = ("import sys; sys.path.insert(0, %r);"
                "from boldt_embed import rag_teacher_scoring;"
                "assert 'torch' not in sys.modules; print('clean')") % str(ROOT / "src")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


if __name__ == "__main__":
    unittest.main()
