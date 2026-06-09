"""Stdlib tests for hybrid retrieval: RRF, BM25, metrics, Matryoshka sweep, dry-run."""
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data_pipeline as dp  # noqa: E402
from boldt_embed import hybrid_eval as H  # noqa: E402

CORPUS = ROOT / "tests" / "fixtures" / "hybrid_corpus.jsonl"
QUERIES = ROOT / "tests" / "fixtures" / "hybrid_queries.jsonl"
QRELS = ROOT / "tests" / "fixtures" / "hybrid_qrels.jsonl"


class TestRRF(unittest.TestCase):
    def test_fusion_orders_by_combined_rank(self):
        fused = H.reciprocal_rank_fusion([["a", "b", "c"], ["c", "b", "a"]])
        self.assertEqual(fused[0], "a")    # a leads (seen first; ties with c)
        self.assertEqual(fused[-1], "b")   # b never top-ranked
        self.assertEqual(set(fused), {"a", "b", "c"})

    def test_single_doc_boost(self):
        fused = H.reciprocal_rank_fusion([["x", "y"], ["x", "z"]])
        self.assertEqual(fused[0], "x")    # x top in both lists


class TestBM25AndModes(unittest.TestCase):
    def setUp(self):
        self.corpus = [{"id": str(r["doc_id"]), "text": r["text"]} for r in dp.stream_jsonl(CORPUS)]
        self.queries = [{"query_id": "q1", "query": "Wie hoch darf die Mietkaution sein?",
                         "positive_ids": {"c1"}},
                        {"query_id": "q2", "query": "Landeshauptstadt Bayern München",
                         "positive_ids": {"c3"}}]
        self.bm25 = H.bm25_rankings_for_queries(self.queries, self.corpus)

    def test_bm25_finds_positive(self):
        self.assertEqual(self.bm25["q1"][0], "c1")
        self.assertEqual(self.bm25["q2"][0], "c3")

    def test_evaluate_mode_bm25_only(self):
        m = H.evaluate_mode(self.queries, self.bm25, {}, "bm25_only")
        self.assertGreater(m["ndcg@10"], 0.9)
        self.assertIn("recall@100", m)
        self.assertIn("map@10", m)
        self.assertIn("pos_in_top_10", m)

    def test_hybrid_falls_back_without_reranker(self):
        # hybrid_rrf_plus_reranker with no rerank_fn returns the fused ranking (no crash)
        ranked = H.fuse_and_rerank(["c1", "c2"], ["c2", "c1"], "hybrid_rrf_plus_reranker",
                                   rerank_fn=None)
        self.assertEqual(set(ranked), {"c1", "c2"})


class TestMatryoshkaSweep(unittest.TestCase):
    def test_sweep_truncates_and_ranks(self):
        query_vecs = {"q1": [1.0, 0.0, 0.0, 0.0]}
        doc_vecs = [("c1", [1.0, 0.0, 0.0, 0.0]), ("c2", [0.0, 1.0, 0.0, 0.0])]
        queries = [{"query_id": "q1", "positive_ids": {"c1"}}]
        sweep = H.matryoshka_sweep(query_vecs, doc_vecs, queries, dims=[4, 2])
        self.assertEqual(set(sweep), {4, 2})
        for dim in (4, 2):
            self.assertAlmostEqual(sweep[dim]["ndcg@10"], 1.0)  # c1 ranks first at both dims

    def test_truncation_changes_ranking(self):
        # full dim: c2 closer; truncated to first coord: c1 closer -> ranking flips
        query_vecs = {"q1": [1.0, 1.0]}
        doc_vecs = [("c1", [1.0, 0.0]), ("c2", [0.3, 1.0])]
        queries = [{"query_id": "q1", "positive_ids": {"c1"}}]
        sweep = H.matryoshka_sweep(query_vecs, doc_vecs, queries, dims=[2, 1])
        self.assertAlmostEqual(sweep[1]["ndcg@10"], 1.0)  # at dim 1, c1 wins


class TestDryRun(unittest.TestCase):
    def test_script_dry_run(self):
        out = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "eval_hybrid_retrieval.py"),
             "--eval-corpus", str(CORPUS), "--eval-queries", str(QUERIES),
             "--qrels", str(QRELS), "--dims", "1024,256,64", "--dry-run"],
            capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("dry-run-ok", out.stdout)
        self.assertIn("bm25_only", out.stdout)


if __name__ == "__main__":
    unittest.main()
