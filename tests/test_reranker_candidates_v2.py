"""Stdlib tests for v2 reranker candidate-list builder. No ML, no network."""
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import negative_mining_2026 as nm  # noqa: E402

SCRIPT = ROOT / "scripts" / "build_reranker_candidates_v2.py"
CANDS = ROOT / "tests" / "fixtures" / "teacher_candidates.jsonl"


def _corpus():
    return {
        "d1": {"text": "Die Mietkaution darf hoechstens das Dreifache betragen.", "domain": "admin"},
        "d2": {"text": "Ein Mietvertrag kann befristet sein.", "domain": "admin"},
        "d3": {"text": "Die Kaution wird nach Auszug zurueckgezahlt.", "domain": "admin"},
        "d4": {"text": "Muenchen ist die Hauptstadt von Bayern.", "domain": "wiki"},
        "d5": {"text": "Das Wetter ist heute sonnig.", "domain": "web"},
    }


def _positives():
    return [{"query_id": "q1", "query": "Wie hoch darf die Mietkaution sein?",
             "doc_id": "d1", "document": "Die Mietkaution darf hoechstens das Dreifache betragen.",
             "domain": "admin", "source": "syn"}]


def _scores():
    return nm.load_teacher_scores([
        {"query_id": "q1", "doc_id": "d1", "reranker_score": 6.9, "embedding_score": 0.9},
        {"query_id": "q1", "doc_id": "d2", "reranker_score": 0.4, "embedding_score": 0.4},
        {"query_id": "q1", "doc_id": "d3", "reranker_score": 6.8, "embedding_score": 0.8},  # false neg
        {"query_id": "q1", "doc_id": "d4", "reranker_score": -2.0, "embedding_score": 0.3},
        {"query_id": "q1", "doc_id": "d5", "reranker_score": -5.0, "embedding_score": 0.1},
    ])


class TestCandidateLists(unittest.TestCase):
    def setUp(self):
        self.merged = nm.merge_candidate_pools(
            ("bm25", {"q1": ["d2", "d3", "d4"]}),
            ("student_dense", {"q1": ["d5", "d3"]}))

    def test_schema_and_false_negative_vetoed(self):
        rows, stats = nm.build_reranker_candidate_lists(
            _positives(), self.merged, _corpus(), _scores(), negatives_per_query=8, margin=0.5)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        for key in ("query_id", "query", "candidates", "positive_doc_ids", "source", "domain"):
            self.assertIn(key, row)
        labels = {c["doc_id"]: c["label"] for c in row["candidates"]}
        self.assertEqual(labels["d1"], 1)                       # positive
        self.assertNotIn("d3", labels)                          # false negative vetoed (6.8 ~ 6.9)
        self.assertEqual(stats["vetoed_false_negatives"], 1)
        for c in row["candidates"]:
            for k in ("doc_id", "document", "label", "teacher_score", "candidate_source", "domain"):
                self.assertIn(k, c)

    def test_mixed_sources(self):
        rows, stats = nm.build_reranker_candidate_lists(
            _positives(), self.merged, _corpus(), _scores(), negatives_per_query=8, margin=0.5)
        srcs = {c["candidate_source"] for c in rows[0]["candidates"]}
        self.assertIn("positive", srcs)
        self.assertTrue({"bm25", "student_dense"} & srcs)       # at least one negative source
        self.assertIn("bm25", stats["candidates_by_source"])

    def test_teacher_medians_separate(self):
        _, stats = nm.build_reranker_candidate_lists(
            _positives(), self.merged, _corpus(), _scores(), negatives_per_query=8, margin=0.5)
        self.assertGreater(stats["pos_teacher_median"], stats["neg_teacher_median"])

    def test_deterministic(self):
        a, _ = nm.build_reranker_candidate_lists(_positives(), self.merged, _corpus(), _scores())
        b, _ = nm.build_reranker_candidate_lists(_positives(), self.merged, _corpus(), _scores())
        self.assertEqual(a, b)


class TestCLI(unittest.TestCase):
    def test_dry_run(self):
        out = subprocess.run(
            [sys.executable, str(SCRIPT), "--candidates", str(CANDS),
             "--corpus", str(CANDS), "--negatives-per-query", "3", "--dry-run"],
            capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("[reranker-lists]", out.stdout)
        self.assertIn("candidates_by_source", out.stdout)


if __name__ == "__main__":
    unittest.main()
