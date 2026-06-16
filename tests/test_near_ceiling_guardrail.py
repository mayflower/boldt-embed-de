"""Stdlib tests for the near-ceiling guardrail builder. No ML."""
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import near_ceiling_guardrail as NC  # noqa: E402


def mklist(qid, pos_rank, n=20, domain="webfaq_heldout", source="bm25"):
    cands = [{"doc_id": f"{qid}-d{i}", "text": f"p{i}", "first_stage_rank": i,
              "first_stage_score": round(20 - i, 3), "candidate_source": source,
              "is_positive": i == pos_rank} for i in range(n)]
    return {"query_id": qid, "query": f"q-{qid}", "domain": domain,
            "positive_doc_ids": [f"{qid}-d{pos_rank}"], "candidates": cands}


class TestSelection(unittest.TestCase):
    def test_selects_only_near_ceiling(self):
        rows = [mklist("nc1", 0), mklist("nc2", 0),        # near-ceiling (pos at rank0)
                mklist("low", 8),                          # fs nDCG too low
                mklist("few", 0, n=10)]                    # < 20 candidates
        res = NC.build(rows, target_size=100)
        sel = {r["query_id"] for r in res["selected"]}
        self.assertEqual(sel, {"nc1", "nc2"})
        self.assertEqual(res["report"]["status"], "pass")

    def test_excludes_public_eval_sources(self):
        rows = [mklist("nc1", 0), mklist("gq1", 0, domain="germanquad_do_not_train"),
                mklist("dt1", 0, source="dt_test")]
        res = NC.build(rows, exclude_sources={"germanquad", "dt_test"}, target_size=100)
        sel = {r["query_id"] for r in res["selected"]}
        self.assertEqual(sel, {"nc1"})
        self.assertGreaterEqual(res["report"]["excluded_public_or_listed_source"], 2)

    def test_deterministic_order_independent(self):
        rows = [mklist(f"nc{i}", 0) for i in range(10)]
        a = [r["query_id"] for r in NC.build(rows, target_size=5)["selected"]]
        b = [r["query_id"] for r in NC.build(list(reversed(rows)), target_size=5)["selected"]]
        self.assertEqual(a, b)

    def test_fails_if_query_overlaps_training(self):
        rows = [mklist("nc1", 0), mklist("nc2", 0)]
        res = NC.build(rows, train_query_ids={"nc1"}, target_size=100)
        self.assertEqual(res["report"]["status"], "fail")
        self.assertEqual(res["report"]["training_overlap"]["overlap_count"], 1)
        self.assertTrue(any("overlap training" in e for e in res["errors"]))

    def test_disjoint_training_passes(self):
        rows = [mklist("nc1", 0), mklist("nc2", 0)]
        res = NC.build(rows, train_query_ids={"other-q"}, target_size=100)
        self.assertEqual(res["report"]["status"], "pass")


class TestReport(unittest.TestCase):
    def test_report_has_required_fields(self):
        res = NC.build([mklist(f"nc{i}", 0) for i in range(3)], target_size=100)
        for k in ("num_selected", "by_domain", "candidate_source_distribution",
                  "first_stage_ndcg_distribution", "oracle_ndcg_distribution", "leakage_check",
                  "training_overlap", "multi_source_fraction"):
            self.assertIn(k, res["report"])

    def test_report_stable(self):
        rows = [mklist(f"nc{i}", 0) for i in range(5)]
        self.assertEqual(NC.build(rows, target_size=3)["report"],
                         NC.build(rows, target_size=3)["report"])

    def test_near_ceiling_definition(self):
        self.assertTrue(NC.is_near_ceiling(NC.list_metrics(mklist("a", 0))))
        self.assertFalse(NC.is_near_ceiling(NC.list_metrics(mklist("b", 8))))      # low fs
        self.assertFalse(NC.is_near_ceiling(NC.list_metrics(mklist("c", 0, n=10))))  # few cands


if __name__ == "__main__":
    unittest.main()
