"""Stdlib tests for catastrophic-drop error analysis. No ML."""
import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import rerank_error_analysis as EA  # noqa: E402


def mk(qid, specs, query="frage wort", positives=()):
    """specs: list of (doc_id, fs_score, rr_score, text, source)."""
    cands = [{"doc_id": d, "text": t, "candidate_source": s, "first_stage_score": fss,
              "reranker_score": rrs, "is_positive": d in positives}
             for (d, fss, rrs, t, s) in specs]
    return {"query_id": qid, "query": query, "positive_doc_ids": list(positives), "candidates": cands}


def big(positive_fs_rank, positive_rr_last=True, n=12):
    """n candidates; positive at the given first-stage rank; reranker sends it last if requested."""
    specs = []
    for i in range(n):
        fss = 100 - i                       # doc i at first-stage rank i
        is_pos = (i == positive_fs_rank)
        rrs = -100.0 if (is_pos and positive_rr_last) else (50 - i)
        specs.append((f"d{i}", fss, rrs, f"text {i} body", "bm25"))
    pos = (f"d{positive_fs_rank}",)
    return mk("q", specs, positives=pos)


class TestClassify(unittest.TestCase):
    def test_positive_demoted_from_top3(self):
        self.assertEqual(EA.classify_error(big(1)), "positive_demoted_from_top1_or_top3")

    def test_lexical_exact_positive_demoted(self):
        self.assertEqual(EA.classify_error(big(0)), "lexical_exact_positive_demoted")

    def test_reranker_promotes_longer_but_wrong(self):
        row = mk("q", [("P", 20, 1.0, "kurz", "bm25"),
                       ("W", 10, 9.0, "viel laenger " * 10, "bm25"),
                       ("x", 5, 0.5, "noch text", "bm25")], positives=("P",))
        # P at fs rank0 but rr rank1 (in top10) -> rule1 skipped; W (non-pos) longer at rr top1
        self.assertEqual(EA.classify_error(row), "reranker_promotes_longer_but_wrong_doc")

    def test_duplicate_confusion(self):
        row = mk("q", [("P", 20, 9.0, "alpha beta gamma delta", "bm25"),
                       ("Dup", 19, 8.0, "alpha beta gamma delta epsilon", "bm25"),
                       ("x", 5, 0.5, "ganz anderes", "bm25")], positives=("P",))
        self.assertEqual(EA.classify_error(row), "duplicate_or_near_duplicate_confusion")

    def test_candidate_source_artifact_multi_source(self):
        row = mk("q", [("P", 20, 1.0, "passage eins", "bm25"),
                       ("W", 10, 9.0, "passage zwei", "dense"),
                       ("x", 5, 0.5, "passage drei vier", "bm25")], positives=("P",))
        self.assertEqual(EA.classify_error(row), "candidate_source_artifact")

    def test_insufficient_first_stage_features(self):
        row = mk("q", [("P", 5.0, 9.0, "passage eins zwei", "bm25"),
                       ("a", 4.9, 1.0, "passage drei", "bm25"),
                       ("b", 4.8, 0.5, "passage vier", "bm25")], positives=("P",))
        # tiny first-stage gap (0.1); P stays near top under rerank -> insufficient features
        self.assertEqual(EA.classify_error(row), "insufficient_first_stage_features")

    def test_query_style_mismatch(self):
        row = mk("q", [("P", 20, 9.0, " ".join(["wort"] * 30), "bm25"),
                       ("a", 10, 1.0, "kurz", "bm25"),
                       ("b", 5, 0.5, "auch kurz", "bm25")], query="x", positives=("P",))
        self.assertEqual(EA.classify_error(row), "query_style_mismatch")

    def test_unknown(self):
        row = mk("q", [("P", 20, 9.0, "drei wort hier", "bm25"),
                       ("a", 10, 1.0, "drei wort dort", "bm25"),
                       ("b", 5, 0.5, "drei wort weg", "bm25")], query="vier fuenf sechs", positives=("P",))
        self.assertEqual(EA.classify_error(row), "unknown")


class TestFixability(unittest.TestCase):
    def test_top1_lock_fixes_demotion(self):
        row = big(0)                          # positive at fs rank0; always-rerank demotes it
        rec = EA.analyze_query(row)
        self.assertIsNotNone(rec)             # it IS a catastrophic drop
        self.assertTrue(rec["fixable_by"]["top1_lock"])
        self.assertTrue(rec["fixable_by"]["top3_lock"])

    def test_analyze_counts_and_fixability(self):
        rep = EA.analyze([big(0), big(1)])
        self.assertEqual(rep["n_catastrophic"], 2)
        self.assertEqual(rep["fixable_counts"]["top1_lock"], 1)   # only the rank-0 positive
        self.assertEqual(rep["fixable_counts"]["top3_lock"], 2)   # both (rank0 + rank1) within top3
        self.assertEqual(rep["fixable_by_any_policy"], 2)
        self.assertGreaterEqual(rep["counts_by_error_type"].get("lexical_exact_positive_demoted", 0), 1)


class TestReport(unittest.TestCase):
    def test_deterministic_and_order_independent(self):
        rows = [big(0), big(1)]
        a = EA.analyze(rows)
        rev = [json.loads(json.dumps(r)) for r in reversed(rows)]
        for r in rev:
            r["candidates"] = list(reversed(r["candidates"]))
        b = EA.analyze(rev)
        self.assertEqual([r["query_id"] for r in a["records"]],
                         [r["query_id"] for r in b["records"]])
        self.assertEqual(a["fixable_counts"], b["fixable_counts"])

    def test_record_has_schema_fields(self):
        rec = EA.analyze_query(big(0))
        for k in ("query_id", "first_stage_ndcg10", "reranked_ndcg10", "delta", "first_stage_top10",
                  "reranked_top10", "positive_doc_ids", "positive_initial_rank",
                  "positive_final_rank", "first_stage_gap_features", "reranker_gap_features",
                  "rank_displacements", "candidate_source_mix", "error_type", "fixable_by"):
            self.assertIn(k, rec)

    def test_markdown_renders(self):
        md = EA.render_markdown(EA.analyze([big(0), big(1)]))
        self.assertIn("catastrophic", md)
        self.assertIn("error type", md.lower())


class TestNoMl(unittest.TestCase):
    def test_no_torch_in_subprocess(self):
        code = ("import sys; sys.path.insert(0, %r)\n"
                "from boldt_embed import rerank_error_analysis as EA\n"
                "row={'query_id':'q','query':'a b','positive_doc_ids':['d0'],'candidates':["
                "{'doc_id':'d%d','text':'t','candidate_source':'bm25','first_stage_score':100-i,"
                "'reranker_score':(-100 if i==0 else 50-i),'is_positive':i==0} for i in range(12)]}\n"
                "EA.analyze([row])\n"
                "assert 'torch' not in sys.modules, 'torch imported'\n"
                "print('OK')\n" % (str(ROOT / "src"), 0)).replace("d%d", "'+str(i)+'")
        # build the row inline robustly
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "from boldt_embed import rerank_error_analysis as EA\n"
            "cands=[{'doc_id':'d'+str(i),'text':'t '+str(i),'candidate_source':'bm25',"
            "'first_stage_score':100-i,'reranker_score':(-100.0 if i==0 else 50-i),"
            "'is_positive':i==0} for i in range(12)]\n"
            "EA.analyze([{'query_id':'q','query':'a b','positive_doc_ids':['d0'],'candidates':cands}])\n"
            "assert 'torch' not in sys.modules, 'torch imported'\n"
            "print('OK')\n" % (str(ROOT / "src"),)
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("OK", r.stdout)


if __name__ == "__main__":
    unittest.main()
