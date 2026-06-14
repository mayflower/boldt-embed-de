"""Tests for realistic fixed RAG candidate-list building (pure stdlib)."""
import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import rag_candidates as RC  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
SCRIPT = ROOT / "scripts" / "build_rag_candidate_lists.py"


def _read(name):
    return [json.loads(l) for l in (FIX / name).read_text("utf-8").splitlines() if l.strip()]


def _corpus():
    return {d["doc_id"]: d for d in _read("rag_corpus.jsonl")}


def _sources():
    bm25 = {str(r["query_id"]): r.get("candidates") for r in _read("bm25_results.jsonl")}
    dense = {str(r["query_id"]): r.get("results") for r in _read("dense_results.jsonl")}
    return {"bm25": bm25, "v3_dense": dense}


class TestMerge(unittest.TestCase):
    def test_deterministic(self):
        q, c, s = _read("rag_queries.jsonl"), _corpus(), _sources()
        r1, rep1 = RC.build_candidate_lists(q, c, s)
        r2, rep2 = RC.build_candidate_lists(q, c, s)
        self.assertEqual(r1, r2)
        self.assertEqual(rep1, rep2)

    def test_positive_preserved(self):
        rows, _ = RC.build_candidate_lists(_read("rag_queries.jsonl"), _corpus(), _sources())
        for r in rows:
            ids = {c["doc_id"] for c in r["candidates"]}
            self.assertTrue(set(r["positive_doc_ids"]) & ids, r["query_id"])

    def test_dedup_by_doc_id_keeps_priority_source(self):
        rows, _ = RC.build_candidate_lists(_read("rag_queries.jsonl"), _corpus(), _sources())
        q1 = next(r for r in rows if r["query_id"] == "q1")
        d1 = [c for c in q1["candidates"] if c["doc_id"] == "d1"]
        self.assertEqual(len(d1), 1)                  # appears once despite bm25 + dense
        self.assertEqual(d1[0]["candidate_source"], "bm25")   # bm25 has merge priority

    def test_dedup_by_text_hash(self):
        corpus = {"a": {"text": "gleicher text hier", "domain": "web"},
                  "b": {"text": "Gleicher   Text hier", "domain": "web"}}  # same text, diff id
        q = [{"query_id": "q", "query": "x", "positive_doc_ids": ["a"], "domain": "web"}]
        rows, _ = RC.build_candidate_lists(q, corpus, {"bm25": {"q": ["a", "b"]}})
        self.assertEqual(len(rows[0]["candidates"]), 1)

    def test_source_distribution_and_report(self):
        _, rep = RC.build_candidate_lists(_read("rag_queries.jsonl"), _corpus(), _sources())
        self.assertIn("bm25", rep["candidate_source_distribution"])
        self.assertIn("v3_dense", rep["candidate_source_distribution"])
        self.assertEqual(rep["positive_in_top_k_rate"], 1.0)
        self.assertIn("faq_real", rep["domains"])


class TestMissingPositive(unittest.TestCase):
    def _setup(self):
        corpus = {"dpos": {"text": "the answer", "domain": "faq_real"},
                  "dn1": {"text": "noise one", "domain": "faq_real"},
                  "dn2": {"text": "noise two", "domain": "faq_real"}}
        q = [{"query_id": "qm", "query": "x", "positive_doc_ids": ["dpos"], "domain": "faq_real"}]
        src = {"bm25": {"qm": ["dn1", "dn2"]}}     # positive NOT surfaced by any source
        return q, corpus, src

    def test_train_skips_and_reports_missing(self):
        q, c, s = self._setup()
        rows, rep = RC.build_candidate_lists(q, c, s, is_eval=False)
        self.assertEqual(rows, [])                  # no positive -> no train list
        self.assertEqual(rep["missing_positive_queries"], ["qm"])

    def test_eval_injects_positive(self):
        q, c, s = self._setup()
        rows, rep = RC.build_candidate_lists(q, c, s, is_eval=True)
        self.assertEqual(len(rows), 1)              # eval list kept (positive injected)
        self.assertIn("qm", rep["injected_positive_queries"])
        ids = {ca["doc_id"] for ca in rows[0]["candidates"]}
        self.assertIn("dpos", ids)
        inj = next(ca for ca in rows[0]["candidates"] if ca["doc_id"] == "dpos")
        self.assertEqual(inj["candidate_source"], "manual")
        self.assertIsNone(inj["label"])             # eval: labels null (positives from qrels)


class TestLabels(unittest.TestCase):
    def test_train_labels_high_precision(self):
        corpus = {"dpos": {"text": "ans", "domain": "faq_real"},
                  "dneg": {"text": "neg", "domain": "faq_real"},
                  "dunc": {"text": "maybe", "domain": "faq_real"}}
        q = [{"query_id": "q", "query": "x", "positive_doc_ids": ["dpos"], "domain": "faq_real"}]
        ts = {("q", "dneg"): 0.5, ("q", "dunc"): 3.0}   # neg clear; dunc uncertain band
        rows, _ = RC.build_candidate_lists(q, corpus, {"bm25": {"q": ["dpos", "dneg", "dunc"]}},
                                           teacher_scores=ts)
        lab = {c["doc_id"]: c["label"] for c in rows[0]["candidates"]}
        self.assertEqual(lab["dpos"], 1)            # gold positive
        self.assertEqual(lab["dneg"], 0)            # teacher clearly low
        self.assertIsNone(lab["dunc"])              # uncertain -> null (not a hard negative)


class TestCli(unittest.TestCase):
    def test_listed_dry_run(self):
        out = subprocess.run(
            [sys.executable, str(SCRIPT), "--queries", str(FIX / "rag_queries.jsonl"),
             "--corpus", str(FIX / "rag_corpus.jsonl"), "--qrels", str(FIX / "rag_qrels.jsonl"),
             "--bm25-results", str(FIX / "bm25_results.jsonl"),
             "--dense-results", str(FIX / "dense_results.jsonl"),
             "--output", "/tmp/_rag_lists.jsonl", "--dry-run"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("dry-run-ok", out.stdout)

    def test_module_is_stdlib_only(self):
        code = ("import sys; sys.path.insert(0, %r);"
                "from boldt_embed import rag_candidates;"
                "assert 'torch' not in sys.modules; print('clean')") % str(ROOT / "src")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


if __name__ == "__main__":
    unittest.main()
