"""Tests for the v6 candidate-union RRF core (stdlib, no ML)."""
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_v6_candidate_union",
                                              ROOT / "scripts" / "build_v6_candidate_union.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


class RrfFuseTests(unittest.TestCase):
    def test_union_and_sources(self):
        fused, score, src = B.rrf_fuse(["a", "b", "c"], ["c", "d", "e"])
        self.assertEqual(set(fused), {"a", "b", "c", "d", "e"})        # union of both retrievers
        self.assertEqual(src["c"], {"bm25", "dense_v6"})               # c retrieved by both
        self.assertEqual(src["a"], {"bm25"})
        self.assertEqual(src["d"], {"dense_v6"})

    def test_doc_in_both_outranks_single_source(self):
        # 'c' is rank-0 in dense and rank-2 in bm25 -> beats 'a' (only bm25 rank-0)
        fused, score, _ = B.rrf_fuse(["a", "b", "c"], ["c", "x", "y"])
        self.assertEqual(fused[0], "c")
        self.assertGreater(score["c"], score["a"])

    def test_list_size_cap_and_determinism(self):
        a = B.rrf_fuse(["a", "b", "c", "d"], ["e", "f", "g", "h"], list_size=3)[0]
        b = B.rrf_fuse(["a", "b", "c", "d"], ["e", "f", "g", "h"], list_size=3)[0]
        self.assertEqual(len(a), 3)
        self.assertEqual(a, b)                                         # deterministic

    def test_empty_dense_falls_back_to_bm25(self):
        fused, _, src = B.rrf_fuse(["a", "b"], [])
        self.assertEqual(fused, ["a", "b"])
        self.assertTrue(all(src[d] == {"bm25"} for d in fused))


if __name__ == "__main__":
    unittest.main()
