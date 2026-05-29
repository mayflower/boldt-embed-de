import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data  # noqa: E402

SAMPLES = ROOT / "data" / "samples"


class TestPII(unittest.TestCase):
    def test_email_detected(self):
        hits = data.find_pii("Bitte melden Sie sich bei max.mustermann@example.de.")
        self.assertTrue(any(k == "email" for k, _ in hits), hits)

    def test_german_iban_detected(self):
        hits = data.find_pii("Konto: DE89 3704 0044 0532 0130 00 bitte überweisen.")
        self.assertTrue(any(k == "iban_de" for k, _ in hits), hits)

    def test_legal_ref_not_flagged_as_pii(self):
        # "§ 543 Abs. 2 Nr. 3 BGB" must NOT trigger PII patterns.
        self.assertEqual(data.find_pii("Gemäß § 543 Abs. 2 Nr. 3 BGB."), [])

    def test_shipped_samples_have_no_pii(self):
        for name in ("toy_triples_de.jsonl", "toy_pairs_de.jsonl"):
            recs = data.load_jsonl(SAMPLES / name)
            self.assertEqual(data.scan_pii(recs), [], name)


class TestSchemasParse(unittest.TestCase):
    def test_schemas_parse_and_have_required(self):
        tp = json.loads((ROOT / "schemas" / "training_pair.schema.json").read_text("utf-8"))
        self.assertEqual(set(tp["required"]), {"query", "positive", "source", "license"})
        br = json.loads((ROOT / "schemas" / "benchmark_result.schema.json").read_text("utf-8"))
        for field in ("command", "commit", "model", "dataset", "split", "metric", "hardware", "output_path"):
            self.assertIn(field, br["required"], field)


if __name__ == "__main__":
    unittest.main()
