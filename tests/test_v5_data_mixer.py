"""Stdlib tests for the v5 RAG data mixer. No ML, no network."""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import v5_data_mixer as M  # noqa: E402

Q = {
    "faq_real": "Wie hoch darf die Mietkaution sein?",
    "qa_passage_non_eval": "Welche Frist gilt fuer den Widerruf eines Kaufvertrags im Internet?",
    "web_nonfaq": "Erklaere den Unterschied zwischen Brutto- und Nettolohn in Deutschland.",
    "long_doc_chunks": ("Das Dokument beschreibt das Verfahren. Welche Schritte sind noetig, "
                        "um einen Reisepass in Bayern zu beantragen, und welche Unterlagen?"),
    "german_stress": "Donaudampfschifffahrtsgesellschaftskapitaen",
    "local_rag": "Wo finde ich die interne Urlaubsrichtlinie?",
}


def row(sid, domain, *, license="CC-BY-4.0", synthetic_query=False,
        eval_only=False, public_benchmark=False, **kw):
    r = {"source_id": sid, "domain": domain, "query": Q[domain],
         "document": f"{domain} Beleg {sid}: Antworttext mit Inhalt.",
         "license": license, "synthetic_query": synthetic_query,
         "eval_only": eval_only, "public_benchmark": public_benchmark}
    r.update(kw)
    return r


def make_rows(per_domain):
    rows = []
    for domain, count in per_domain.items():
        for i in range(count):
            rows.append(row(f"{domain}-{i}", domain,
                            synthetic_query=(domain in ("web_nonfaq", "german_stress"))))
    return rows


class TestShareGates(unittest.TestCase):
    def test_over_faq_mixture_fails(self):
        rows = make_rows({"faq_real": 50, "web_nonfaq": 50})  # 2 domains -> round-robin 50/50
        rep = M.mix(rows, target_count=10, max_faq_share=0.35, min_nonfaq_share=0.50)
        self.assertEqual(rep["status"], "fail")
        self.assertAlmostEqual(rep["faq_share"], 0.5)
        self.assertTrue(any("FAQ share" in e for e in rep["errors"]))

    def test_balanced_mixture_passes(self):
        rows = make_rows({"faq_real": 10, "qa_passage_non_eval": 10, "web_nonfaq": 10,
                          "long_doc_chunks": 10, "german_stress": 10})
        rep = M.mix(rows, target_count=10, max_faq_share=0.35, min_nonfaq_share=0.50)
        self.assertEqual(rep["status"], "pass", rep["errors"])
        self.assertLessEqual(rep["faq_share"], 0.35)
        self.assertGreaterEqual(rep["nonfaq_share"], 0.50)
        # report proves non-FAQ coverage across multiple domains
        self.assertGreaterEqual(len([d for d in rep["rows_by_domain"] if d != "faq_real"]), 3)

    def test_too_few_nonfaq_fails_min_share(self):
        # plenty FAQ, a sliver of one non-FAQ domain -> non-FAQ share collapses below floor
        rows = make_rows({"faq_real": 100, "web_nonfaq": 1})
        rep = M.mix(rows, target_count=20, max_faq_share=0.95, min_nonfaq_share=0.50)
        self.assertEqual(rep["status"], "fail")
        self.assertTrue(any("non-FAQ share" in e for e in rep["errors"]))


class TestHardFails(unittest.TestCase):
    def test_public_benchmark_flag_leakage_fails(self):
        rows = make_rows({"faq_real": 5, "web_nonfaq": 5})
        rows.append(row("leak-1", "qa_passage_non_eval", public_benchmark=True))
        rep = M.mix(rows, target_count=10, max_faq_share=0.9, min_nonfaq_share=0.0)
        self.assertEqual(rep["status"], "fail")
        self.assertTrue(any("leakage" in e for e in rep["errors"]))

    def test_eval_only_leakage_fails(self):
        rows = make_rows({"web_nonfaq": 5})
        rows.append(row("leak-2", "qa_passage_non_eval", eval_only=True))
        rep = M.mix(rows, target_count=10, max_faq_share=1.0, min_nonfaq_share=0.0)
        self.assertEqual(rep["status"], "fail")

    def test_public_benchmark_token_in_source_fails(self):
        rows = make_rows({"web_nonfaq": 5})
        rows.append(row("germanquad-test-split", "qa_passage_non_eval"))
        rep = M.mix(rows, target_count=10, max_faq_share=1.0, min_nonfaq_share=0.0)
        self.assertEqual(rep["status"], "fail")
        self.assertTrue(any("leakage" in e for e in rep["errors"]))

    def test_unknown_license_fails(self):
        rows = make_rows({"web_nonfaq": 5})
        rows.append(row("bad-lic", "qa_passage_non_eval", license="proprietary-internal"))
        rep = M.mix(rows, target_count=10, max_faq_share=1.0, min_nonfaq_share=0.0)
        self.assertEqual(rep["status"], "fail")
        self.assertTrue(any("license" in e for e in rep["errors"]))

    def test_known_inherited_synthetic_license_is_allowed(self):
        rows = make_rows({"web_nonfaq": 4, "qa_passage_non_eval": 4})
        rows.append(row("ok-lic", "web_nonfaq", license="synthetic-inherits-source",
                        synthetic_query=True))
        rep = M.mix(rows, target_count=9, max_faq_share=1.0, min_nonfaq_share=0.0)
        self.assertEqual(rep["status"], "pass", rep["errors"])


class TestDeterminism(unittest.TestCase):
    def test_sampling_is_deterministic_and_order_independent(self):
        rows = make_rows({"faq_real": 8, "qa_passage_non_eval": 8, "web_nonfaq": 8,
                          "long_doc_chunks": 8, "german_stress": 8})
        a = M.mix(rows, target_count=15, max_faq_share=0.35, min_nonfaq_share=0.50)
        b = M.mix(list(reversed(rows)), target_count=15, max_faq_share=0.35, min_nonfaq_share=0.50)
        ids_a = [r["source_id"] for r in a["selected"]]
        ids_b = [r["source_id"] for r in b["selected"]]
        self.assertEqual(ids_a, ids_b)            # identical even when input order differs
        self.assertEqual(a["rows_by_domain"], b["rows_by_domain"])

    def test_report_has_required_sections(self):
        rows = make_rows({"faq_real": 6, "qa_passage_non_eval": 6, "web_nonfaq": 6,
                          "long_doc_chunks": 6, "german_stress": 6})
        rep = M.mix(rows, target_count=10, max_faq_share=0.35, min_nonfaq_share=0.50)
        for k in ("rows_by_domain", "rows_by_source", "rows_by_license", "faq_share",
                  "nonfaq_share", "synthetic_share", "real_share",
                  "query_style_distribution", "examples_per_domain"):
            self.assertIn(k, rep)
        self.assertTrue(rep["examples_per_domain"])  # examples per domain present


if __name__ == "__main__":
    unittest.main()
