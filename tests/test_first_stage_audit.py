"""Tests for the first-stage recall audit (stdlib, no ML)."""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import first_stage_audit as FA  # noqa: E402


def _cand(doc_id, src, rank, score=None):
    return {"doc_id": doc_id, "candidate_source": src, "first_stage_rank": rank,
            "first_stage_score": score if score is not None else (None if rank is None else 20 - rank),
            "text": f"text-{doc_id}"}


def _row(qid, positives, cands, domain="faq"):
    return {"query_id": qid, "positive_doc_ids": list(positives), "domain": domain,
            "query": f"q-{qid}", "candidates": cands}


class ClassifyTests(unittest.TestCase):
    def test_source_classification(self):
        self.assertEqual(FA.classify_source("bm25"), "bm25")
        self.assertEqual(FA.classify_source("manual"), "injected")
        self.assertEqual(FA.classify_source("dense-boldt-v5"), "dense")
        self.assertEqual(FA.classify_source("e5-base"), "dense")
        self.assertEqual(FA.classify_source(""), "other")


class MissingPositiveTests(unittest.TestCase):
    def test_injected_only_positive_is_counted_missing(self):
        # positive present only via 'manual' (first_stage_rank None) -> retriever never retrieved it
        row = _row("q1", ["p1"], [_cand("p1", "manual", None), _cand("d2", "bm25", 0),
                                  _cand("d3", "bm25", 1)])
        q = FA.audit_query(row)
        self.assertEqual(q["missing_count"], 1)
        self.assertEqual(q["missing_positives"], ["p1"])
        self.assertEqual(q["pos_injected_only"], ["p1"])
        self.assertEqual(q["recall@10"], 0.0)
        self.assertFalse(q["positive_in_top_10"])

    def test_absent_positive_is_counted_missing(self):
        row = _row("q1", ["pX"], [_cand("d1", "bm25", 0), _cand("d2", "bm25", 1)])
        q = FA.audit_query(row)
        self.assertEqual(q["missing_count"], 1)
        self.assertEqual(q["pos_absent"], ["pX"])

    def test_retrieved_positive_is_not_missing(self):
        row = _row("q1", ["p1"], [_cand("p1", "bm25", 0), _cand("d2", "bm25", 1)])
        q = FA.audit_query(row)
        self.assertEqual(q["missing_count"], 0)
        self.assertEqual(q["recall@10"], 1.0)
        self.assertTrue(q["positive_in_top_10"])

    def test_set_missing_count_aggregates(self):
        rows = [
            _row("q1", ["p1"], [_cand("p1", "manual", None), _cand("d2", "bm25", 0)]),   # missing
            _row("q2", ["p2"], [_cand("p2", "bm25", 0), _cand("d3", "bm25", 1)]),         # retrieved
            _row("q3", ["pZ"], [_cand("d4", "bm25", 0)]),                                 # absent
        ]
        rep = FA.audit_set(rows, name="t")
        self.assertEqual(rep["missing_positive_count"], 2)
        self.assertAlmostEqual(rep["missing_positive_rate"], 2 / 3, places=4)
        self.assertEqual(rep["bottleneck"]["primary"], "first_stage_recall")


class SourceContributionTests(unittest.TestCase):
    def test_bm25_only_dense_only_overlap_union(self):
        rows = [
            _row("q1", ["p1"], [_cand("p1", "bm25", 0)]),                 # bm25-only
            _row("q2", ["p2"], [_cand("p2", "dense-v5", 0)]),             # dense-only
            _row("q3", ["p3a", "p3b"], [_cand("p3a", "bm25", 0),
                                        _cand("p3b", "dense-v5", 1)]),    # one bm25, one dense
        ]
        c = FA.audit_set(rows, name="t")["candidate_source_contribution"]
        self.assertEqual(c["bm25_hits"], 2)         # p1, p3a
        self.assertEqual(c["dense_hits"], 2)        # p2, p3b
        self.assertEqual(c["bm25_only"], 2)
        self.assertEqual(c["dense_only"], 2)
        self.assertEqual(c["overlap_bm25_and_dense"], 0)
        self.assertEqual(c["union_any_retriever"], 4)
        self.assertTrue(c["has_dense_source"])

    def test_overlap_counts_when_positive_in_both(self):
        # a positive that appears with BOTH a bm25 and a dense candidate row (same doc_id)
        row = _row("q1", ["p1"], [_cand("p1", "bm25", 0), _cand("p1", "dense-v5", 0),
                                  _cand("d2", "bm25", 1)])
        c = FA.audit_set([row], name="t")["candidate_source_contribution"]
        self.assertEqual(c["overlap_bm25_and_dense"], 1)
        self.assertEqual(c["bm25_only"], 0)
        self.assertEqual(c["dense_only"], 0)
        self.assertEqual(c["union_any_retriever"], 1)

    def test_bm25_only_lists_report_no_dense(self):
        rows = [_row("q1", ["p1"], [_cand("p1", "manual", None), _cand("d2", "bm25", 0)])]
        c = FA.audit_set(rows, name="t")["candidate_source_contribution"]
        self.assertFalse(c["has_dense_source"])


class OracleUpperBoundTests(unittest.TestCase):
    def test_oracle_retriever_zero_when_positive_injected_only(self):
        # realistic ceiling must NOT credit an injected positive
        row = _row("q1", ["p1"], [_cand("p1", "manual", None), _cand("d2", "bm25", 0),
                                  _cand("d3", "bm25", 1)])
        q = FA.audit_query(row)
        self.assertEqual(q["oracle_ndcg10_retriever"], 0.0)        # cannot be reranked into place
        self.assertGreater(q["oracle_ndcg10_with_injected"], 0.0)  # only if injection counted
        self.assertEqual(q["upper_bound_reranker_lift"], 0.0)
        self.assertGreater(q["illusory_lift_from_injection"], 0.0)

    def test_oracle_retriever_full_when_positive_retrieved_low(self):
        # positive retrieved but ranked low -> realistic reranker headroom is positive
        cands = [_cand(f"d{i}", "bm25", i) for i in range(5)] + [_cand("p1", "bm25", 5)]
        row = _row("q1", ["p1"], cands)
        q = FA.audit_query(row)
        self.assertEqual(q["oracle_ndcg10_retriever"], 1.0)
        self.assertLess(q["first_stage_ndcg10"], 1.0)
        self.assertGreater(q["upper_bound_reranker_lift"], 0.0)

    def test_no_reranker_claim_if_positive_absent(self):
        # acceptance: no reranker lift may be claimed for an unretrieved positive
        rows = [_row("q1", ["pX"], [_cand("d1", "bm25", 0)])]
        rep = FA.audit_set(rows, name="t")
        self.assertEqual(rep["upper_bound_reranker_lift_realistic"], 0.0)
        self.assertEqual(rep["oracle_ndcg10_retriever"], 0.0)


class DeterminismTests(unittest.TestCase):
    def test_examples_are_deterministic(self):
        rows = [_row(f"q{i}", [f"p{i}"], [_cand(f"p{i}", "manual", None), _cand(f"d{i}", "bm25", 0)])
                for i in range(20)]
        a = FA.audit_set(rows, name="t", max_examples=5)["examples"]
        b = FA.audit_set(rows, name="t", max_examples=5)["examples"]
        self.assertEqual([e["query_id"] for e in a], [e["query_id"] for e in b])
        self.assertEqual(len(a), 5)


if __name__ == "__main__":
    unittest.main()
