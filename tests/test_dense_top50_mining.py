"""Tests for v6.1 dense top-50 hard-negative mining (stdlib, no ML)."""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import dense_top50_mining as M  # noqa: E402


def _qrec(qid, pos_rank, n=210, domain="faq_real", **extra):
    docs = [f"{qid}-d{i}" for i in range(n)]
    pid = f"{qid}-pos"
    docs[pos_rank - 1] = pid                       # place positive at 1-indexed pos_rank
    r = {"query_id": qid, "query": f"q {qid}", "positive_doc_id": pid, "domain": domain,
         "source": "webfaq_train", "dense_ranked": docs}
    r.update(extra)
    return r


def _corpus(*qrecs):
    return {d: f"text {d}" for q in qrecs for d in q["dense_ranked"]}


class TargetSelectionTests(unittest.TestCase):
    def test_detects_rank_51_100_case(self):
        q = _qrec("a", 73)
        rec = M.mine_query(q, _corpus(q))
        self.assertIsNotNone(rec)
        self.assertEqual(rec["positive_rank_v6"], 73)
        self.assertEqual(rec["positive_doc_id"], "a-pos")

    def test_skips_when_positive_in_top50(self):
        q = _qrec("a", 20)
        self.assertIsNone(M.mine_query(q, _corpus(q)))

    def test_skips_beyond_window(self):
        q = _qrec("a", 205, n=210)
        self.assertIsNone(M.mine_query(q, _corpus(q), window=200))

    def test_skips_when_positive_absent(self):
        q = _qrec("a", 73)
        q["dense_ranked"] = [d for d in q["dense_ranked"] if d != "a-pos"]   # remove positive
        self.assertIsNone(M.mine_query(q, _corpus(q)))


class NegativeMiningTests(unittest.TestCase):
    def test_mines_negatives_above_positive(self):
        q = _qrec("a", 73)
        rec = M.mine_query(q, _corpus(q), max_negatives=30)
        self.assertTrue(rec["negatives"])
        for ng in rec["negatives"]:
            self.assertLess(ng["negative_rank_v6"], 73)        # all ranked ABOVE the positive
            self.assertEqual(ng["source"], M.SRC_DENSE_FP)
            self.assertNotEqual(ng["doc_id"], "a-pos")
        # highest blockers first, capped
        self.assertEqual(rec["negatives"][0]["negative_rank_v6"], 1)
        self.assertLessEqual(len(rec["negatives"]), 30)

    def test_cap_respected(self):
        q = _qrec("a", 73)
        self.assertEqual(len(M.mine_query(q, _corpus(q), max_negatives=10)["negatives"]), 10)

    def test_false_negative_veto(self):
        q = _qrec("a", 73)
        # FP at rank 5 is teacher-close to the positive (veto); FP at rank 10 is clearly worse (keep)
        ts = {"a-pos": 8.0, "a-d4": 7.5, "a-d9": 1.0}    # ranks 5 and 10 (0-indexed d4,d9)
        rec = M.mine_query(q, _corpus(q), teacher_scores=ts, veto_margin=2.0, max_negatives=200)
        ids = {n["doc_id"] for n in rec["negatives"]}
        self.assertNotIn("a-d4", ids)                  # vetoed (margin 0.5 < 2.0)
        self.assertIn("a-d9", ids)                      # kept (margin 7.0)
        self.assertGreaterEqual(rec["_vetoed"], 1)
        kept = next(n for n in rec["negatives"] if n["doc_id"] == "a-d9")
        self.assertEqual(kept["margin_to_positive"], 7.0)

    def test_teacher_source_below_positive(self):
        # a teacher-confirmed hard neg BELOW the positive (rank 90) becomes a 'teacher' negative
        q = _qrec("a", 73)
        ts = {"a-pos": 8.0, "a-d89": 0.0}              # rank 90, margin 8.0 >= veto
        rec = M.mine_query(q, _corpus(q), teacher_scores=ts, max_negatives=200)
        teacher_negs = [n for n in rec["negatives"] if n["source"] == M.SRC_TEACHER]
        self.assertTrue(any(n["doc_id"] == "a-d89" for n in teacher_negs))


class DeterminismLeakageTests(unittest.TestCase):
    def test_deterministic(self):
        q = _qrec("a", 80)
        a = M.mine_query(q, _corpus(q))
        b = M.mine_query(q, _corpus(q))
        self.assertEqual([n["doc_id"] for n in a["negatives"]],
                         [n["doc_id"] for n in b["negatives"]])

    def test_no_public_eval_train_leakage(self):
        good = _qrec("a", 73)
        leak = _qrec("gq1", 73, domain="germanquad")
        leak["public_benchmark"] = True
        out = M.mine_set([good, leak], _corpus(good, leak))
        ids = {r["query_id"] for r in out["records"]}
        self.assertIn("a", ids)
        self.assertNotIn("gq1", ids)                    # public-eval query excluded
        self.assertGreaterEqual(out["report"]["leakage_excluded"], 1)

    def test_report_rank_buckets_and_veto(self):
        q1 = _qrec("a", 60)        # 51-100
        q2 = _qrec("b", 150)       # 101-200
        out = M.mine_set([q1, q2], _corpus(q1, q2))
        self.assertEqual(out["report"]["positive_rank_51_100"], 1)
        self.assertEqual(out["report"]["positive_rank_101_200"], 1)
        self.assertEqual(out["report"]["queries_mined"], 2)
        for r in out["records"]:                        # internal field stripped from output
            self.assertNotIn("_vetoed", r)


if __name__ == "__main__":
    unittest.main()
