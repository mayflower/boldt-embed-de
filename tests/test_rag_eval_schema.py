"""Tests for RAG-reranker eval schemas, leakage-safe splits, and metrics. Pure stdlib."""
import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import rag_eval_schema as R  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
VALIDATE = ROOT / "scripts" / "validate_rag_eval_sets.py"


def _read(name):
    return [json.loads(l) for l in (FIX / name).read_text("utf-8").splitlines() if l.strip()]


class TestSchema(unittest.TestCase):
    def test_valid_fixtures(self):
        errs = R.validate_eval_set(_read("rag_queries.jsonl"), _read("rag_corpus.jsonl"),
                                   _read("rag_qrels.jsonl"))
        self.assertEqual(errs, [])

    def test_query_schema_failures(self):
        self.assertTrue(R.validate_rag_query({"query": "q"}))                     # no id/positives
        self.assertTrue(R.validate_rag_query({"query_id": "q", "query": "q", "domain": "d",
                                              "source": "s", "positive_doc_ids": [],
                                              }))                                  # empty positives
        self.assertTrue(R.validate_rag_query({"query_id": "q", "query": "q", "domain": "d",
                                              "source": "s", "positive_doc_ids": ["d1"],
                                              "metadata": {"answer_type": "nope"}}))  # bad enum

    def test_corpus_schema_failures(self):
        self.assertTrue(R.validate_rag_corpus_doc({"doc_id": "d", "text": "t"}))  # no source/domain/license
        self.assertEqual(R.validate_rag_corpus_doc(
            {"doc_id": "d", "text": "t", "source": "s", "domain": "faq_real", "license": "CC-BY-4.0"}), [])

    def test_positive_not_in_corpus_fails(self):
        q = [{"query_id": "q1", "query": "x", "domain": "faq_real", "source": "s",
              "positive_doc_ids": ["MISSING"]}]
        c = [{"doc_id": "d1", "text": "t", "source": "s", "domain": "faq_real", "license": "CC-BY-4.0"}]
        errs = R.validate_eval_set(q, c)
        self.assertTrue(any("not in corpus" in e for e in errs), errs)


class TestCandidateList(unittest.TestCase):
    def test_missing_positive_fails(self):
        cl = {"query_id": "q1", "query": "x", "positive_doc_ids": ["dpos"],
              "candidates": [{"doc_id": "dneg1", "text": "a", "candidate_source": "bm25", "label": 0},
                             {"doc_id": "dneg2", "text": "b", "candidate_source": "dense", "label": 0}]}
        errs = R.validate_candidate_list(cl, require_positive=True)
        self.assertTrue(any("at least one positive" in e for e in errs), errs)

    def test_present_positive_passes(self):
        cl = {"query_id": "q1", "query": "x", "positive_doc_ids": ["dpos"],
              "candidates": [{"doc_id": "dpos", "text": "a", "candidate_source": "bm25", "label": 1},
                             {"doc_id": "dneg", "text": "b", "candidate_source": "dense", "label": 0}]}
        self.assertEqual(R.validate_candidate_list(cl, require_positive=True), [])

    def test_bad_candidate_source_and_label(self):
        cl = {"query_id": "q1", "query": "x", "positive_doc_ids": ["d"],
              "candidates": [{"doc_id": "d", "text": "a", "candidate_source": "magic", "label": 2}]}
        errs = R.validate_candidate_list(cl)
        self.assertTrue(any("candidate_source" in e for e in errs))
        self.assertTrue(any("label must be" in e for e in errs))


class TestSplitAndLeakage(unittest.TestCase):
    def test_deterministic_split(self):
        for key in ("q_alpha", "q_beta", "frage 123"):
            self.assertEqual(R.assign_split(key), R.assign_split(key))   # stable
        self.assertIn(R.assign_split("anything"), {"train", "dev", "test"})

    def test_webfaq_split_partitions_without_overlap(self):
        rows = [{"query": f"Frage Nummer {i}?", "answer": f"Antwort {i} mit genug Text hier."}
                for i in range(400)]
        splits = {s: R.build_webfaq_eval(rows, split=s) for s in ("train", "dev", "test")}
        qids = {s: {q["query_id"] for q in splits[s][1]} for s in splits}
        self.assertGreater(len(qids["test"]), 0)
        self.assertGreater(len(qids["dev"]), 0)
        # no query_id shared across splits (leakage-safe)
        self.assertEqual(qids["train"] & qids["test"], set())
        self.assertEqual(qids["train"] & qids["dev"], set())
        self.assertEqual(qids["dev"] & qids["test"], set())

    def test_public_eval_leakage_into_train_fails(self):
        eval_q, eval_d = {"qE"}, {"dE"}
        train = [{"query_id": "qT", "doc_id": "dT", "candidates": [{"doc_id": "dE"}]},  # leaks dE
                 {"query_id": "qE", "doc_id": "dT"}]                                     # leaks qE
        bad = R.check_no_eval_leakage(train, eval_q, eval_d)
        self.assertEqual(len(bad), 2, bad)
        # clean training set -> no leakage
        self.assertEqual(R.check_no_eval_leakage(
            [{"query_id": "qT", "doc_id": "dT", "candidates": [{"doc_id": "dT2"}]}], eval_q, eval_d), [])


class TestMetrics(unittest.TestCase):
    def test_rag_metrics_and_answer_support(self):
        m = R.rag_metrics_for_query(["dpos", "d2", "d3"], {"dpos"}, requires_answer_support=True)
        self.assertEqual(m["ndcg@10"], 1.0)
        self.assertEqual(m["positive_in_top_10"], 1.0)
        self.assertEqual(m["answer_support_at_10"], 1.0)
        # non-answer-support query has no answer_support_at_10 key
        m2 = R.rag_metrics_for_query(["d2", "dpos"], {"dpos"}, requires_answer_support=False)
        self.assertNotIn("answer_support_at_10", m2)

    def test_aggregate_answer_support_only_over_applicable(self):
        rows = [R.rag_metrics_for_query(["dpos"], {"dpos"}, True),
                R.rag_metrics_for_query(["dneg"], {"dpos"}, True),   # miss
                R.rag_metrics_for_query(["dneg"], {"dpos"}, False)]  # not answer-support
        agg = R.aggregate_rag(rows)
        self.assertEqual(agg["answer_support_queries"], 2)
        self.assertEqual(agg["answer_support_at_10"], 0.5)          # 1 of 2 answer-support hit

    def test_reranker_delta(self):
        # first stage ranks the positive last; reranker promotes it to the top -> positive delta
        d = R.reranker_delta_ndcg10(["n1", "n2", "dpos"], ["dpos", "n1", "n2"], {"dpos"})
        self.assertGreater(d["reranker_delta_ndcg10"], 0.0)
        self.assertEqual(d["reranked_ndcg@10"], 1.0)


class TestCli(unittest.TestCase):
    def test_validate_cli_ok(self):
        out = subprocess.run([sys.executable, str(VALIDATE),
                              "--queries", str(FIX / "rag_queries.jsonl"),
                              "--corpus", str(FIX / "rag_corpus.jsonl"),
                              "--qrels", str(FIX / "rag_qrels.jsonl")], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)

    def test_module_is_stdlib_only(self):
        code = ("import sys; sys.path.insert(0, %r);"
                "from boldt_embed import rag_eval_schema;"
                "assert 'torch' not in sys.modules; print('clean')") % str(ROOT / "src")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


if __name__ == "__main__":
    unittest.main()
