"""Tests for the build-once BM25 index (src/boldt_embed/bm25_index.py) and its use in mining.

Pure stdlib. Covers deterministic ranking, save/load round-trip, batch==individual, that the
index is built ONCE (not per query), German ß/ss + umlaut-fold orthography, and the explicit
mining-cap / full-corpus gate.
"""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from boldt_embed import bm25_index as bi  # noqa: E402
from boldt_embed import negative_mining_2026 as nm  # noqa: E402

FIX = ROOT / "tests" / "fixtures"


def _corpus():
    return [json.loads(l) for l in (FIX / "bm25_corpus.jsonl").read_text("utf-8").splitlines()]


class TestRanking(unittest.TestCase):
    def test_deterministic_ranking(self):
        idx = bi.build_bm25_index(_corpus())
        r1 = idx.search("Wie hoch darf die Mietkaution sein?", 10)
        r2 = idx.search("Wie hoch darf die Mietkaution sein?", 10)
        self.assertEqual(r1, r2)                 # identical across calls
        self.assertEqual(r1[0][0], "d1")         # the relevant doc ranks first
        # scores strictly non-increasing
        self.assertEqual([s for _, s in r1], sorted([s for _, s in r1], reverse=True))

    def test_tie_break_is_doc_id(self):
        idx = bi.build_bm25_index([{"doc_id": "b", "text": "katze hund"},
                                   {"doc_id": "a", "text": "katze hund"}])
        # equal scores -> doc_id ascending
        self.assertEqual([d for d, _ in idx.search("katze hund", 10)], ["a", "b"])

    def test_save_load_roundtrip(self):
        idx = bi.build_bm25_index(_corpus())
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            path = f.name
        idx.save(path)
        loaded = bi.BM25Index.load(path)
        for q in ("Mietkaution", "Donau Stadt", "Photosynthese Pflanzen"):
            self.assertEqual(idx.search(q, 10), loaded.search(q, 10))

    def test_batch_equals_individual(self):
        idx = bi.build_bm25_index(_corpus())
        qs = ["Mietkaution Hoehe", "Kuendigungsfrist", "Donau", "Bundestag Wahl"]
        self.assertEqual(idx.batch_search(qs, 5), [idx.search(q, 5) for q in qs])


class TestBuildOnce(unittest.TestCase):
    def test_index_built_once_for_many_queries(self):
        corpus = _corpus()
        queries = [{"query_id": f"q{i}", "query": "Mietkaution Donau Photosynthese"} for i in range(50)]
        before = bi.BM25Index._BUILD_COUNT
        nm.mine_bm25_candidates(queries, corpus, k=5)         # no prebuilt index passed
        self.assertEqual(bi.BM25Index._BUILD_COUNT - before, 1, "must build the index exactly once")

    def test_prebuilt_index_not_rebuilt(self):
        corpus = _corpus()
        idx = bi.build_bm25_index(corpus)
        before = bi.BM25Index._BUILD_COUNT
        nm.mine_bm25_candidates([{"query_id": "q", "query": "Donau"}], corpus, k=5, index=idx)
        idx.batch_search(["a", "b", "c"], 5)
        self.assertEqual(bi.BM25Index._BUILD_COUNT, before, "passing an index must not rebuild")


class TestGermanOrthography(unittest.TestCase):
    def test_eszett_folds_to_ss(self):
        idx = bi.build_bm25_index([{"doc_id": "x", "text": "Die Bahnhofstraße ist gesperrt"},
                                   {"doc_id": "y", "text": "Ein voellig anderer Satz"}])
        # query uses 'strasse' (ss); doc has 'straße' (ß) -> must match via ß->ss fold
        res = idx.search("bahnhofstrasse", 5)
        self.assertTrue(res and res[0][0] == "x")

    def test_umlaut_fold_optional(self):
        docs = [{"doc_id": "m", "text": "Gruesse aus München"}]
        folded = bi.build_bm25_index(docs, fold_umlauts=True)
        self.assertTrue(folded.search("muenchen", 5))          # ü->ue makes 'muenchen' match
        plain = bi.build_bm25_index(docs, fold_umlauts=False)
        self.assertEqual(plain.search("muenchen", 5), [])      # without folding, no match


class TestMiningCapGate(unittest.TestCase):
    SCRIPT = ROOT / "scripts" / "mine_hard_negatives_2026.py"

    def _run(self, *extra):
        return subprocess.run(
            [sys.executable, str(self.SCRIPT),
             "--candidates", str(FIX / "candidates_v2_tiny.jsonl"),
             "--teacher-cache", str(FIX / "teacher_cache_v2_tiny.jsonl"),
             "--output", "/tmp/_hn.jsonl", "--dry-run", *extra],
            capture_output=True, text=True)

    def test_full_mode_default(self):
        out = self._run()
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn('"mining_cap_applied": false', out.stdout)

    def test_require_full_corpus_fails_when_capped(self):
        out = self._run("--max-queries", "2", "--require-full-corpus")
        self.assertEqual(out.returncode, 2)
        self.assertIn("require-full-corpus", out.stderr)

    def test_max_queries_sets_cap_flag(self):
        out = self._run("--max-queries", "2")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn('"mining_cap_applied": true', out.stdout)


class TestStdlibOnly(unittest.TestCase):
    def test_module_is_stdlib_only(self):
        code = ("import sys; sys.path.insert(0, %r);"
                "from boldt_embed import bm25_index;"
                "assert 'torch' not in sys.modules; print('clean')") % str(ROOT / "src")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


if __name__ == "__main__":
    unittest.main()
