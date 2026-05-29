import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import hard_negatives as hn  # noqa: E402
from boldt_embed.textutil import normalize  # noqa: E402

SPECS = ROOT / "data" / "synthetic" / "prompt_specs.json"


class TestOutcomeFlip(unittest.TestCase):
    def test_outcome_flip_changes_result(self):
        text = "Eine fristlose Kündigung ist bei erheblichem Mietrückstand möglich."
        neg = hn.neg_outcome_flip(text)
        self.assertIsNotNone(neg)
        self.assertIn("ausgeschlossen", neg)
        self.assertNotEqual(normalize(neg), normalize(text))

    def test_outcome_flip_registered(self):
        self.assertIn("outcome_flip", hn.GENERATORS)

    def test_seven_families(self):
        self.assertEqual(len(hn.GENERATORS), 7)


class TestQueryTypes(unittest.TestCase):
    def test_ten_query_types(self):
        self.assertEqual(len(hn.QUERY_TYPES), 10)

    def test_templates_cover_all_types(self):
        specs = json.loads(SPECS.read_text(encoding="utf-8"))
        templates = specs["query_type_templates"]
        self.assertEqual(set(templates), set(hn.QUERY_TYPES))
        for t in templates.values():
            self.assertIn("{passage}", t)


if __name__ == "__main__":
    unittest.main()
