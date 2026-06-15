"""Stdlib tests for hardness-aware RAG eval. No ML, no network."""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import hardness_aware_eval as H  # noqa: E402


def pqm(bucket, delta, qid="q", fs=0.6, oracle=0.9):
    return {"query_id": qid, "hardness_bucket": bucket, "delta": delta,
            "first_stage_ndcg@10": fs, "reranked_ndcg@10": fs + delta, "oracle_ndcg@10": oracle,
            "positive_in_top_10": 1.0, "positive_in_top_50": 1.0,
            "num_candidates": 20, "num_candidate_sources": 2}


def clist(qid, positive_rank, ncands=10):
    cands = []
    for i in range(1, ncands + 1):
        cands.append({"doc_id": f"{qid}-d{i}", "first_stage_rank": i,
                      "reranker_score": 100.0 if i == positive_rank else float(ncands - i),
                      "is_positive": (i == positive_rank),
                      "source": "bm25" if i % 2 else "dense", "text": f"doc {i}"})
    return {"query_id": qid, "candidates": cands}


class TestBucketAssignment(unittest.TestCase):
    def test_all_buckets(self):
        self.assertEqual(H.assign_bucket(1.0, 1.0), "no_room")
        self.assertEqual(H.assign_bucket(0.96, 0.99), "no_room")
        self.assertEqual(H.assign_bucket(0.90, 0.99), "easy")     # fs>=0.85 but <0.95 -> not no_room
        self.assertEqual(H.assign_bucket(0.85, 0.90), "easy")
        self.assertEqual(H.assign_bucket(0.70, 0.90), "medium")
        self.assertEqual(H.assign_bucket(0.50, 0.85), "medium")
        self.assertEqual(H.assign_bucket(0.30, 0.90), "hard")     # fs<0.5, oracle>=0.8
        self.assertEqual(H.assign_bucket(0.30, 0.70), "impossible")
        self.assertEqual(H.assign_bucket(0.99, 0.70), "impossible")  # oracle<0.8 dominates
        self.assertEqual(H.assign_bucket(0.40, 0.79), "impossible")


class TestListMetrics(unittest.TestCase):
    def test_medium_list_lifts(self):
        m = H.list_metrics(clist("a", positive_rank=3))
        self.assertEqual(m["hardness_bucket"], "medium")          # fs=0.5, oracle=1.0
        self.assertAlmostEqual(m["first_stage_ndcg@10"], 0.5, 3)
        self.assertAlmostEqual(m["reranked_ndcg@10"], 1.0, 6)     # reranker puts positive first
        self.assertGreater(m["delta"], 0.0)
        self.assertEqual(m["num_candidate_sources"], 2)

    def test_no_room_list_no_change(self):
        m = H.list_metrics(clist("b", positive_rank=1))
        self.assertEqual(m["hardness_bucket"], "no_room")         # fs=1.0, oracle=1.0
        self.assertAlmostEqual(m["delta"], 0.0, 6)

    def test_positive_absent_is_impossible(self):
        row = {"query_id": "c", "positive_doc_ids": ["not-present"],
               "candidates": [{"doc_id": f"c-d{i}", "first_stage_rank": i, "source": "bm25"}
                              for i in range(1, 6)]}
        m = H.list_metrics(row)
        self.assertEqual(m["hardness_bucket"], "impossible")      # oracle 0.0


class TestAggregation(unittest.TestCase):
    def test_primary_uses_only_medium_and_hard(self):
        pq = [pqm("no_room", 0.5, "n"), pqm("easy", 0.5, "e"),   # excluded from primary
              pqm("medium", 0.10, "m"), pqm("hard", 0.20, "h"),
              pqm("impossible", 0.5, "i")]                        # excluded from primary
        s = H.summarize_eval_set("webfaq", pq, role="primary")
        self.assertEqual(s["primary_n"], 2)
        self.assertAlmostEqual(s["primary_micro_lift"], 0.15, 6)  # mean(0.10, 0.20)
        self.assertAlmostEqual(s["primary_macro_lift"], 0.15, 6)  # mean(medium=0.10, hard=0.20)

    def test_catastrophic_detection(self):
        pq = [pqm("medium", 0.1, "a"), pqm("medium", -0.30, "b"), pqm("hard", -0.25, "c"),
              pqm("medium", 0.05, "d")]
        s = H.summarize_eval_set("local_rag", pq, role="primary")
        self.assertEqual(s["catastrophic_drops"], 2)              # -0.30 and -0.25 <= -0.2
        self.assertAlmostEqual(s["catastrophic_rate"], 0.5, 6)
        self.assertIn("b", s["catastrophic_query_ids"])

    def test_mostly_no_room(self):
        pq = [pqm("no_room", 0.0, "1"), pqm("no_room", -0.001, "2"),
              pqm("no_room", 0.0, "3"), pqm("medium", 0.1, "4")]
        s = H.summarize_eval_set("germanquad", pq, role="guardrail")
        self.assertAlmostEqual(s["no_room_fraction"], 0.75, 6)
        self.assertTrue(s["mostly_no_room"])


class TestGate(unittest.TestCase):
    def _primary(self, deltas, name="webfaq"):
        pq = [pqm("medium" if i % 2 else "hard", d, f"{name}-{i}") for i, d in enumerate(deltas)]
        return H.summarize_eval_set(name, pq, role="primary")

    def _guardrail_no_room(self, overall_delta, name="germanquad", catastrophic=0):
        pq = [pqm("no_room", overall_delta, f"{name}-{i}", fs=0.99, oracle=1.0) for i in range(10)]
        for j in range(catastrophic):
            pq[j] = pqm("no_room", -0.3, f"{name}-c{j}", fs=0.99, oracle=1.0)
        return H.summarize_eval_set(name, pq, role="guardrail")

    def test_gate_pass(self):
        sets = [self._primary([0.10, 0.20, 0.15]), self._guardrail_no_room(-0.003)]
        g = H.evaluate_hardness_gate(sets)
        self.assertEqual(g["status"], "pass", g["failing"])

    def test_gate_fail_guardrail_regression(self):
        sets = [self._primary([0.10, 0.20]), self._guardrail_no_room(-0.02)]  # beyond -0.005 tol
        g = H.evaluate_hardness_gate(sets)
        self.assertEqual(g["status"], "fail")
        self.assertTrue(any("guardrail" in c["check"] for c in g["failing"]))

    def test_near_ceiling_tolerance(self):
        # within tolerance passes, just beyond fails
        self.assertEqual(H.evaluate_hardness_gate(
            [self._primary([0.1, 0.2]), self._guardrail_no_room(-0.004)])["status"], "pass")
        self.assertEqual(H.evaluate_hardness_gate(
            [self._primary([0.1, 0.2]), self._guardrail_no_room(-0.010)])["status"], "fail")

    def test_gate_fail_primary_not_positive(self):
        sets = [self._primary([-0.01, 0.0]), self._guardrail_no_room(0.0)]
        g = H.evaluate_hardness_gate(sets)
        self.assertEqual(g["status"], "fail")
        self.assertTrue(any("primary_lift" in c["check"] for c in g["failing"]))

    def test_gate_fail_catastrophic_rate(self):
        # primary lift positive overall but too many per-query catastrophic drops
        pq = [pqm("medium", 0.3, f"x{i}") for i in range(5)] + \
             [pqm("medium", -0.3, f"c{i}") for i in range(5)]
        primary = H.summarize_eval_set("webfaq", pq, role="primary")
        g = H.evaluate_hardness_gate([primary, self._guardrail_no_room(0.0)])
        self.assertEqual(g["status"], "fail")
        self.assertTrue(any("catastrophic" in c["check"] for c in g["failing"]))

    def test_gate_fail_primary_has_no_medium_hard(self):
        only_ceiling = H.summarize_eval_set(
            "webfaq", [pqm("no_room", 0.0, f"n{i}", fs=0.99, oracle=1.0) for i in range(4)],
            role="primary")
        g = H.evaluate_hardness_gate([only_ceiling])
        self.assertEqual(g["status"], "fail")          # no medium+hard => cannot establish lift

    def test_gate_needs_a_primary_set(self):
        g = H.evaluate_hardness_gate([self._guardrail_no_room(0.0)])
        self.assertEqual(g["status"], "fail")


if __name__ == "__main__":
    unittest.main()
