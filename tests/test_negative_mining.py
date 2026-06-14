"""Stdlib tests for teacher-driven hard-negative mining."""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import negative_mining_2026 as nm  # noqa: E402


def _corpus_lookup():
    return {
        "d1": {"text": "Die Mietkaution darf höchstens das Dreifache betragen.", "domain": "admin"},
        "d2": {"text": "Ein Mietvertrag kann befristet sein.", "domain": "admin"},
        "d3": {"text": "Die Kaution wird nach Auszug zurückgezahlt.", "domain": "admin"},
        "d4": {"text": "München ist die Hauptstadt von Bayern.", "domain": "wiki"},
        "d5": {"text": "Das Wetter ist heute sonnig.", "domain": "web"},
    }


def _positives():
    return [{"query_id": "q1", "query": "Wie hoch darf die Mietkaution sein?",
             "doc_id": "d1", "document": "Die Mietkaution darf höchstens das Dreifache betragen.",
             "domain": "admin", "source": "syn"}]


class TestPools(unittest.TestCase):
    def test_bm25_candidates_topk(self):
        corpus = [{"id": k, "text": v["text"]} for k, v in _corpus_lookup().items()]
        res = nm.mine_bm25_candidates(
            [{"query_id": "q1", "query": "Mietkaution Kaution"}], corpus, k=3)
        # BM25Index returns only LEXICAL matches (no zero-score padding): d1 ("Mietkaution")
        # and d3 ("Kaution") match; d2/d4/d5 do not.
        self.assertLessEqual(len(res["q1"]), 3)
        self.assertIn("d1", res["q1"])
        self.assertIn("d3", res["q1"])

    def test_dense_candidates(self):
        qe = {"q1": [1.0, 0.0]}
        de = [("d1", [1.0, 0.0]), ("d2", [0.0, 1.0]), ("d3", [0.9, 0.1])]
        res = nm.mine_dense_candidates_from_embeddings(qe, de, k=2)
        self.assertEqual(res["q1"][0], "d1")  # most similar first

    def test_merge_dedup_and_source(self):
        merged = nm.merge_candidate_pools(
            ("bm25", {"q1": ["d2", "d3"]}),
            ("dense", {"q1": ["d3", "d4"]}))
        ids = [c["doc_id"] for c in merged["q1"]]
        self.assertEqual(ids, ["d2", "d3", "d4"])  # d3 dedup, order preserved
        srcs = {c["doc_id"]: c["source"] for c in merged["q1"]}
        self.assertEqual(srcs["d3"], "bm25")  # first source wins


class TestFalseNegativeFilter(unittest.TestCase):
    def test_reason_logic(self):
        self.assertEqual(nm.false_negative_reason(0.9, 0.95, 0.1), "neg_score_ge_positive")
        self.assertEqual(nm.false_negative_reason(0.9, 0.85, 0.1), "within_margin_of_positive")
        self.assertIsNone(nm.false_negative_reason(0.9, 0.5, 0.1))
        self.assertIsNone(nm.false_negative_reason(None, 0.5, 0.1))  # can't judge -> keep

    def test_filter_drops_false_negatives(self):
        negs = [
            {"doc_id": "d2", "_scores": {"reranker_score": 0.2, "embedding_score": None}},
            {"doc_id": "d3", "_scores": {"reranker_score": 0.95, "embedding_score": None}},  # ~ positive
        ]
        kept, dropped = nm.filter_false_negatives(0.9, negs, margin=0.1)
        self.assertEqual([k["doc_id"] for k in kept], ["d2"])
        self.assertEqual(dropped.get("neg_score_ge_positive"), 1)
        self.assertIsNone(kept[0]["false_negative_filter_reason"])


class TestBuildRows(unittest.TestCase):
    def setUp(self):
        self.merged = nm.merge_candidate_pools(
            ("bm25", {"q1": ["d2", "d3", "d4", "d5"]}))
        # positive d1 high; d3 close (false neg); d2/d4/d5 progressively lower
        self.scores = nm.load_teacher_scores([
            {"query_id": "q1", "doc_id": "d1", "reranker_score": 0.90, "embedding_score": 0.9},
            {"query_id": "q1", "doc_id": "d2", "reranker_score": 0.40, "embedding_score": 0.4},
            {"query_id": "q1", "doc_id": "d3", "reranker_score": 0.88, "embedding_score": 0.8},
            {"query_id": "q1", "doc_id": "d4", "reranker_score": 0.30, "embedding_score": 0.3},
            {"query_id": "q1", "doc_id": "d5", "reranker_score": 0.10, "embedding_score": 0.1},
        ])

    def test_schema_and_false_negative_dropped(self):
        rows, stats = nm.build_triplets_or_lists(
            _positives(), self.merged, _corpus_lookup(), self.scores,
            negatives_per_query=8, margin=0.1)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        for key in ("query_id", "query", "positive_doc_id", "positive", "negatives",
                    "source", "domain"):
            self.assertIn(key, row)
        neg_ids = [n["doc_id"] for n in row["negatives"]]
        self.assertNotIn("d3", neg_ids)  # false negative (0.88 within 0.1 of 0.90) dropped
        self.assertNotIn("d1", neg_ids)  # positive never a negative
        for n in row["negatives"]:
            for key in ("doc_id", "document", "source", "domain",
                        "embedding_teacher_score", "reranker_teacher_score",
                        "false_negative_filter_reason"):
                self.assertIn(key, n)
        self.assertEqual(stats["dropped_by_reason"].get("within_margin_of_positive"), 1)

    def test_hardest_first_ordering(self):
        rows, _ = nm.build_triplets_or_lists(
            _positives(), self.merged, _corpus_lookup(), self.scores,
            negatives_per_query=8, margin=0.1)
        ids = [n["doc_id"] for n in rows[0]["negatives"]]
        self.assertEqual(ids, ["d2", "d4", "d5"])  # 0.40 > 0.30 > 0.10

    def test_deterministic(self):
        a, _ = nm.build_triplets_or_lists(_positives(), self.merged, _corpus_lookup(),
                                          self.scores, negatives_per_query=8, margin=0.1)
        b, _ = nm.build_triplets_or_lists(_positives(), self.merged, _corpus_lookup(),
                                          self.scores, negatives_per_query=8, margin=0.1)
        self.assertEqual(a, b)

    def test_domain_balance_cap(self):
        rows, _ = nm.build_triplets_or_lists(
            _positives(), self.merged, _corpus_lookup(), self.scores,
            negatives_per_query=8, margin=0.1, max_per_domain=1)
        doms = [n["domain"] for n in rows[0]["negatives"]]
        self.assertEqual(doms.count("admin"), 1)  # d2 and (dropped d3) admin -> capped to 1


class TestNoTeacherKeepsAll(unittest.TestCase):
    def test_without_scores_nothing_dropped(self):
        merged = nm.merge_candidate_pools(("bm25", {"q1": ["d2", "d3", "d4"]}))
        rows, stats = nm.build_triplets_or_lists(_positives(), merged, _corpus_lookup(),
                                                 teacher_scores={}, negatives_per_query=8)
        self.assertEqual(stats["dropped_by_reason"], {})
        self.assertEqual(len(rows[0]["negatives"]), 3)


if __name__ == "__main__":
    unittest.main()
