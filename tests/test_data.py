import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data  # noqa: E402

SAMPLES = ROOT / "data" / "samples"


class TestToySamplesValid(unittest.TestCase):
    def test_triples_validate_clean(self):
        records = data.load_jsonl(SAMPLES / "toy_triples_de.jsonl")
        report = data.validate_dataset(records)
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.num_records, 7)
        self.assertEqual(report.num_with_negatives, 7)
        # every German hard-negative family is represented at least once
        for family in (
            "compound", "negation", "legal_ref",
            "dates_numbers", "regional_variant", "entity_disambiguation",
        ):
            self.assertIn(family, report.neg_type_counts, family)

    def test_pairs_validate_clean(self):
        records = data.load_jsonl(SAMPLES / "toy_pairs_de.jsonl")
        report = data.validate_dataset(records)
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.num_records, 6)
        self.assertEqual(report.num_with_negatives, 0)

    def test_shipped_samples_have_allowed_licenses(self):
        for name in ("toy_triples_de.jsonl", "toy_pairs_de.jsonl"):
            records = data.load_jsonl(SAMPLES / name)
            self.assertEqual(data.check_licenses(records), [], name)


class TestRecordValidation(unittest.TestCase):
    def test_missing_positive_flagged(self):
        errs = data.validate_record({"query": "q", "source": "s", "license": "synthetic"}, 0)
        self.assertTrue(any("positive" in e for e in errs), errs)

    def test_disallowed_license_flagged(self):
        errs = data.validate_record(
            {"query": "q", "positive": "p", "source": "s", "license": "GPL-3.0"}, 3
        )
        self.assertTrue(any("disallowed license" in e for e in errs), errs)

    def test_negative_equal_to_positive_flagged(self):
        errs = data.validate_record(
            {"query": "q", "positive": "Berlin ist die Hauptstadt.",
             "negatives": ["berlin ist die hauptstadt."],  # equal after normalization
             "source": "s", "license": "synthetic"},
            1,
        )
        self.assertTrue(any("identical to the positive" in e for e in errs), errs)

    def test_unknown_neg_type_flagged(self):
        errs = data.validate_record(
            {"query": "q", "positive": "p", "negatives": ["n"],
             "neg_types": ["bogus"], "source": "s", "license": "synthetic"},
            2,
        )
        self.assertTrue(any("unknown neg_type" in e for e in errs), errs)


class TestLeakage(unittest.TestCase):
    def test_exact_leakage_detected(self):
        records = [{"query": "q", "positive": "Berlin ist die Hauptstadt Deutschlands.",
                    "source": "s", "license": "synthetic"}]
        eval_texts = ["Berlin ist die Hauptstadt Deutschlands."]
        hits = data.find_leakage(records, eval_texts)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["kind"], "exact")

    def test_near_duplicate_leakage_detected(self):
        records = [{"query": "q",
                    "positive": "Die Widerrufsfrist beträgt 14 Tage ab Erhalt der Ware.",
                    "source": "s", "license": "synthetic"}]
        eval_texts = ["Die Widerrufsfrist beträgt 14 Tage ab dem Erhalt der Ware."]
        hits = data.find_leakage(records, eval_texts, threshold=0.8)
        self.assertTrue(hits and hits[0]["kind"] in {"exact", "near_dup"}, hits)

    def test_clean_data_has_no_leakage(self):
        records = data.load_jsonl(SAMPLES / "toy_triples_de.jsonl")
        eval_texts = ["Ein völlig anderes Thema über Astronomie und Sternbilder am Nachthimmel."]
        self.assertEqual(data.find_leakage(records, eval_texts), [])


if __name__ == "__main__":
    unittest.main()
