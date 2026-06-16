"""Stdlib tests for the bounded policy serving wrapper. No ML, no labels at inference."""
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import policy_reranker as PR  # noqa: E402
from boldt_embed.policy_config import load_policy  # noqa: E402

POLICY = load_policy(ROOT / "configs" / "policies" / "bounded_margin_override_v1.json")


def mklist(qid, rr_scores, n=8):
    cands = [{"doc_id": f"{qid}-d{i}", "text": f"p{i}", "first_stage_rank": i,
              "first_stage_score": round(20 - 1.5 * i, 3), "candidate_source": "bm25",
              "reranker_score": rr_scores[i]} for i in range(n)]
    return {"query_id": qid, "query": f"q-{qid}", "candidates": cands}


def fs_order(out):
    return [c["doc_id"] for c in sorted(out["candidates"], key=lambda c: c["first_stage_rank"])]


def final_order(out):
    return [c["doc_id"] for c in sorted(out["candidates"], key=lambda c: c["final_rank"])]


class TestBounds(unittest.TestCase):
    def test_top_k_lock_respected(self):
        # no margin override (fs_top1 reranker score near top)
        out = PR.rerank_query(mklist("a", [9, 5, 4, 3, 2, 1, 0.5, 0.2]), POLICY)
        k = POLICY["bounds"]["preserve_first_stage_top_k"]
        self.assertEqual(final_order(out)[:k], [f"a-d{i}" for i in range(k)])
        self.assertEqual(out["diagnostics"]["top_k_locked"], k)
        self.assertFalse(out["diagnostics"]["margin_override_used"])
        self.assertTrue(all(c["policy_action"] == "locked"
                            for c in out["candidates"] if c["final_rank"] < k))

    def test_max_downshift_respected(self):
        out = PR.rerank_query(mklist("b", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 9.0]), POLICY)
        md = POLICY["bounds"]["max_downshift"]
        for c in out["candidates"]:
            self.assertLessEqual(c["final_rank"] - c["first_stage_rank"], md,
                                 f"{c['doc_id']} dropped > {md}")
        self.assertLessEqual(out["diagnostics"]["max_downshift"], md)

    def test_margin_override_works(self):
        # doc at fs rank5 beats fs_top1 reranker score by >= margin (3.0)
        out = PR.rerank_query(mklist("c", [1.0, 2, 1.5, 1.2, 0.8, 9.0, 0.4, 0.3]), POLICY)
        self.assertTrue(out["diagnostics"]["margin_override_used"])
        self.assertEqual(final_order(out)[0], "c-d5")
        top = [c for c in out["candidates"] if c["final_rank"] == 0][0]
        self.assertEqual(top["policy_action"], "margin_override")


class TestModes(unittest.TestCase):
    def test_raw_rerank_forbidden_by_default(self):
        with self.assertRaises(ValueError):
            PR.rerank_query(mklist("d", [1, 9, 2, 3, 4, 5, 6, 7]), POLICY, mode="raw_rerank")

    def test_raw_rerank_with_flag_works(self):
        out = PR.rerank_query(mklist("d", [1, 9, 2, 3, 4, 5, 6, 7]), POLICY, mode="raw_rerank",
                              allow_raw=True)
        self.assertEqual(final_order(out)[0], "d-d1")   # highest reranker score
        self.assertTrue(all(c["policy_action"] == "reranked" for c in out["candidates"]))

    def test_first_stage_only(self):
        out = PR.rerank_query(mklist("e", [0.1, 9, 8, 7, 6, 5, 4, 3]), POLICY, mode="first_stage_only")
        self.assertEqual(final_order(out), fs_order(out))
        self.assertTrue(all(c["policy_action"] == "kept_first_stage" for c in out["candidates"]))


class TestSafety(unittest.TestCase):
    def test_qrels_ignored_even_if_present(self):
        row = mklist("f", [1.0, 2, 1.5, 1.2, 0.8, 9.0, 0.4, 0.3])
        leaked = json.loads(json.dumps(row))
        leaked["positive_doc_ids"] = ["f-d0"]
        for c in leaked["candidates"]:
            c["is_positive"] = c["doc_id"] == "f-d0"
            c["label"] = 1
        a = PR.rerank_query(row, POLICY)
        b = PR.rerank_query(leaked, POLICY)
        self.assertEqual(final_order(a), final_order(b))
        self.assertEqual([c["policy_action"] for c in a["candidates"]],
                         [c["policy_action"] for c in b["candidates"]])

    def test_deterministic_order_independent(self):
        row = mklist("g", [1.0, 2, 1.5, 1.2, 0.8, 9.0, 0.4, 0.3])
        rev = json.loads(json.dumps(row)); rev["candidates"] = list(reversed(rev["candidates"]))
        self.assertEqual(final_order(PR.rerank_query(row, POLICY)),
                         final_order(PR.rerank_query(rev, POLICY)))


class TestMalformed(unittest.TestCase):
    def test_missing_candidates_fails(self):
        with self.assertRaises(ValueError):
            PR.rerank_query({"query_id": "x"}, POLICY)

    def test_candidate_missing_doc_id_fails(self):
        with self.assertRaises(ValueError):
            PR.rerank_query({"query_id": "x", "candidates": [{"first_stage_rank": 0}]}, POLICY)

    def test_candidate_missing_first_stage_fails(self):
        with self.assertRaises(ValueError):
            PR.rerank_query({"query_id": "x", "candidates": [{"doc_id": "d0", "reranker_score": 1}]},
                            POLICY)


if __name__ == "__main__":
    unittest.main()
