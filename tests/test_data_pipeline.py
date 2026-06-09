"""Stdlib tests for the multi-domain data pipeline + German adversarial generator."""
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data_pipeline as dp  # noqa: E402
from boldt_embed import german_adversarial as ga  # noqa: E402

SEEDS = ROOT / "tests" / "fixtures" / "adversarial_seeds.jsonl"


class TestHashesAndIds(unittest.TestCase):
    def test_text_hash_deterministic_and_normalizing(self):
        self.assertEqual(dp.stable_text_hash("Hallo Welt"), dp.stable_text_hash("Hallo Welt"))
        # whitespace/NFC differences collapse to the same hash
        self.assertEqual(dp.stable_text_hash("Hallo   Welt"), dp.stable_text_hash("Hallo Welt"))
        self.assertNotEqual(dp.stable_text_hash("Hallo"), dp.stable_text_hash("Welt"))

    def test_pair_id_deterministic(self):
        a = dp.stable_pair_id("q", "d")
        b = dp.stable_pair_id("q", "d")
        self.assertEqual(a, b)
        self.assertTrue(a.startswith("p"))

    def test_language_hint(self):
        self.assertEqual(dp.detect_language_hint_simple("Die Mietkaution ist für den Mieter"), "de")
        self.assertEqual(dp.detect_language_hint_simple("Größe"), "de")  # umlaut/ß
        self.assertEqual(dp.detect_language_hint_simple("the quick brown fox"), "unknown")


class TestNormalizeValidate(unittest.TestCase):
    def test_normalize_fills_ids_and_defaults(self):
        r = dp.normalize_record({"query": "  Wie   geht  es? ", "document": "Gut."},
                                default_source="syn", default_domain="faq", default_license="CC0")
        self.assertEqual(r["query"], "Wie geht es?")
        self.assertTrue(r["query_id"].startswith("q"))
        self.assertTrue(r["doc_id"].startswith("d"))
        self.assertTrue(r["positive"])
        self.assertEqual(r["source"], "syn")
        self.assertEqual(dp.validate_candidate_record(r), [])

    def test_validate_flags_missing_and_bad_positive(self):
        errs = dp.validate_candidate_record({"query_id": "q", "doc_id": "d", "query": "x",
                                             "document": "y", "source": "s", "domain": "d",
                                             "license": "l", "positive": "yes"})
        self.assertTrue(any("positive" in e for e in errs), errs)


class TestSelectionFilters(unittest.TestCase):
    def _rows(self):
        return [
            dp.normalize_record({"query": "a", "document": "doc1", "domain": "web", "source": "s", "license": "l"}),
            dp.normalize_record({"query": "a", "document": "doc1", "domain": "web", "source": "s", "license": "l"}),
            dp.normalize_record({"query": "b", "document": "doc2", "domain": "web", "source": "s", "license": "l"}),
            dp.normalize_record({"query": "c", "document": "doc3", "domain": "faq", "source": "s", "license": "l"}),
        ]

    def test_dedup(self):
        rows = dp.deduplicate_by_text_hash(self._rows())
        self.assertEqual(len(rows), 3)  # first (a,doc1) duplicate removed

    def test_domain_balance(self):
        rows = dp.domain_balanced_sample(self._rows(), max_per_domain=1)
        doms = [r["domain"] for r in rows]
        self.assertEqual(doms.count("web"), 1)
        self.assertEqual(doms.count("faq"), 1)

    def test_leakage_filter(self):
        rows = self._rows()
        kept, stats = dp.filter_leakage_against_eval_texts(rows, eval_texts=["doc2"])
        self.assertEqual(stats["dropped"], 1)
        self.assertTrue(all(r["document"] != "doc2" for r in kept))


class TestAdversarial(unittest.TestCase):
    def setUp(self):
        self.seeds = list(dp.stream_jsonl(SEEDS))

    def test_deterministic(self):
        a = ga.generate_adversarial_candidates(self.seeds)
        b = ga.generate_adversarial_candidates(self.seeds)
        self.assertEqual(a, b)

    def test_marks_source_and_domain(self):
        rows = ga.generate_adversarial_candidates(self.seeds)
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(r["source"], "synthetic_adversarial")
            self.assertEqual(r["domain"], "german_stress")
            self.assertEqual(dp.validate_candidate_record(r), [])
            self.assertIn("template_id", r["metadata"])

    def test_paraphrase_positive_distractor_negative(self):
        rows = ga.generate_adversarial_candidates(self.seeds, emit_anchor=False)
        by_t = {}
        for r in rows:
            by_t.setdefault(r["metadata"]["template_id"], r["positive"])
        # orthographic/register/legal-wording variants are positive
        self.assertTrue(by_t.get("ss_eszett", True))
        # negation / number / date / wrong-section / entity swaps are hard negatives
        for neg in ("negation", "number_shift", "date_shift", "entity_swap"):
            if neg in by_t:
                self.assertFalse(by_t[neg], f"{neg} should be a distractor (positive=False)")

    def test_eszett_transform(self):
        self.assertEqual(ga.swap_eszett("Straße"), "Strasse")
        self.assertIsNone(ga.swap_eszett("Haus"))

    def test_number_change_is_distractor(self):
        self.assertEqual(ga.change_number("genau 3 Monate"), "genau 4 Monate")


class TestScriptsDryRun(unittest.TestCase):
    def test_build_candidates_dry_run_no_ml(self):
        code = (
            "import sys; sys.argv=['build_training_candidates.py','--source-jsonl',%r,"
            "'--default-source','syn','--default-domain','faq','--default-license','CC0','--dry-run'];"
            "sys.path.insert(0, %r);"
            "import build_training_candidates as b; rc=b.main(); assert rc==0, rc;"
            "assert 'torch' not in sys.modules; print('OK')"
        ) % (str(SEEDS), str(ROOT / "scripts"))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("OK", out.stdout)

    def test_generate_adversarial_dry_run(self):
        out = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_german_adversarial.py"),
             "--input", str(SEEDS), "--dry-run"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("DRY RUN", out.stdout)
        self.assertIn("synthetic_adversarial", out.stdout)


if __name__ == "__main__":
    unittest.main()
