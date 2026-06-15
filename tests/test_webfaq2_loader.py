"""Stdlib tests for the WebFAQ2 hard-negative loader. No network, no ML."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import webfaq2_loader as W  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "webfaq2_de_tiny.jsonl"


def _records():
    return W.load_local_jsonl(str(FIXTURE))


class TestLocalImport(unittest.TestCase):
    def test_local_fixture_import(self):
        out = W.import_webfaq2(_records(), language="de")
        rep = out["report"]
        self.assertEqual(rep["status"], "pass")
        self.assertEqual(rep["imported_queries"], 1)        # the en record is skipped
        self.assertEqual(rep["skipped_other_language"], 1)
        # margins for q1: 7.0 (keep), 2.5 (keep), 0.2 (false-neg), 1.5 (insufficient)
        self.assertEqual(rep["embedder_triplets"], 2)
        self.assertEqual(rep["reranker_lists"], 1)
        self.assertEqual(rep["dropped_false_negatives"], 1)
        self.assertEqual(rep["dropped_insufficient_margin"], 1)

    def test_reranker_list_has_listwise_scores(self):
        out = W.import_webfaq2(_records(), language="de")
        lst = out["reranker_lists"][0]
        self.assertEqual(len(lst["candidates"]), 3)         # 1 positive + 2 kept negatives
        pos = [c for c in lst["candidates"] if c["is_positive"]]
        self.assertEqual(len(pos), 1)
        self.assertEqual(pos[0]["doc_id"], lst["positive_doc_id"])
        self.assertAlmostEqual(sum(c["teacher_softmax_target"] for c in lst["candidates"]), 1.0, 5)

    def test_embedder_triplets_carry_teacher_margin(self):
        out = W.import_webfaq2(_records(), language="de")
        for t in out["triplets"]:
            self.assertAlmostEqual(t["teacher_margin"], t["positive_score"] - t["negative_score"], 5)
            self.assertGreaterEqual(t["teacher_margin"], 2.0)   # all kept negatives clear the margin


class TestFilters(unittest.TestCase):
    def _rec(self, neg_scores, pos=8.0, license="CC-BY-4.0", language="de"):
        return {"query": "Frage?", "positive": "Antwort.", "positive_score": pos,
                "language": language, "license": license,
                "negatives": [{"document": f"neg {i}", "cross_encoder_score": s}
                              for i, s in enumerate(neg_scores)]}

    def test_margin_filter_keeps_only_clear_negatives(self):
        out = W.import_webfaq2([self._rec([1.0, 6.5, 5.9])], language="de", min_margin=2.0,
                               false_negative_margin=0.5)
        # margins 7.0 keep, 1.5 insufficient, 2.1 keep
        self.assertEqual(out["report"]["negatives_per_query"]["total"], 2)
        self.assertEqual(out["report"]["dropped_insufficient_margin"], 1)

    def test_false_negative_drop(self):
        out = W.import_webfaq2([self._rec([7.8, 8.1, 1.0])], language="de", min_margin=2.0,
                               false_negative_margin=0.5)
        # margins 0.2 and -0.1 are false negatives; 7.0 kept
        self.assertEqual(out["report"]["dropped_false_negatives"], 2)
        self.assertEqual(out["report"]["negatives_per_query"]["total"], 1)

    def test_max_negatives_cap_is_deterministic(self):
        rec = self._rec([0.0, 0.5, 1.0, 1.5, 2.0])   # margins 8..6, all >= 2.0
        a = W.import_webfaq2([rec], language="de", max_negatives=2)
        b = W.import_webfaq2([rec], language="de", max_negatives=2)
        self.assertEqual(a["report"]["negatives_per_query"]["total"], 2)
        self.assertEqual([t["negative"] for t in a["triplets"]],
                         [t["negative"] for t in b["triplets"]])      # deterministic

    def test_unknown_license_fails_closed(self):
        out = W.import_webfaq2([self._rec([1.0], license="")], language="de")
        self.assertEqual(out["report"]["status"], "fail")
        self.assertTrue(any("license" in e for e in out["report"]["errors"]))

    def test_missing_positive_score_fails(self):
        rec = {"query": "Q?", "positive": "A.", "language": "de", "license": "CC-BY-4.0",
               "negatives": [{"document": "n", "cross_encoder_score": 1.0}]}
        out = W.import_webfaq2([rec], language="de")
        self.assertEqual(out["report"]["status"], "fail")


class TestPropagationAndShapes(unittest.TestCase):
    def test_license_and_language_propagation(self):
        out = W.import_webfaq2(_records(), language="de")
        for t in out["triplets"]:
            self.assertEqual(t["license"], "CC-BY-4.0")
            self.assertEqual(t["language"], "de")
        self.assertEqual(out["reranker_lists"][0]["license"], "CC-BY-4.0")
        self.assertEqual(out["report"]["by_license"], {"cc-by-4.0": 1})

    def test_string_negatives_with_parallel_scores(self):
        rec = {"query": "Q?", "positive": "A.", "positive_score": 8.0, "language": "de",
               "license": "CC-BY-4.0", "negatives": ["n1", "n2"],
               "negative_scores": [1.0, 5.0]}
        out = W.import_webfaq2([rec], language="de")
        self.assertEqual(out["report"]["status"], "pass")
        self.assertEqual(out["report"]["negatives_per_query"]["total"], 2)


class TestDryRunNoNetwork(unittest.TestCase):
    def test_cli_dry_run_no_network(self):
        with tempfile.TemporaryDirectory() as d:
            report = pathlib.Path(d) / "report.json"
            output = pathlib.Path(d) / "triplets.jsonl"
            r = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "import_webfaq2_hardnegatives.py"),
                 "--input", str(FIXTURE), "--output", str(output), "--report", str(report),
                 "--language", "de", "--min-cross-encoder-margin", "2.0",
                 "--max-negatives-per-query", "32", "--dry-run"],
                capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("dry-run-ok", r.stdout)
            self.assertTrue(report.exists())
            self.assertFalse(output.exists())          # dry-run writes no data file
            rep = json.loads(report.read_text("utf-8"))
            self.assertEqual(rep["imported_queries"], 1)


if __name__ == "__main__":
    unittest.main()
