import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import hard_negatives as hn  # noqa: E402
from boldt_embed.textutil import normalize  # noqa: E402


class TestGenerators(unittest.TestCase):
    cases = {
        "compound": "Die Kündigungsfrist für eine Mietwohnung beträgt drei Monate.",
        "negation": "Bei diesem Vertrag besteht ein gesetzliches Widerrufsrecht.",
        "legal_ref": "Gemäß § 543 Abs. 2 Nr. 3 BGB ist eine fristlose Kündigung möglich.",
        "dates_numbers": "Die Widerrufsfrist beträgt 14 Tage ab Erhalt der Ware.",
        "regional_variant": "In Österreich heißt der erste Monat Jänner.",
        "entity_disambiguation": "Der VW Golf ist ein Auto des Herstellers Volkswagen.",
    }

    def test_each_generator_changes_text(self):
        for cat, text in self.cases.items():
            neg = hn.GENERATORS[cat](text)
            self.assertIsNotNone(neg, f"{cat} returned None")
            self.assertNotEqual(normalize(neg), normalize(text), cat)

    def test_legal_ref_changes_paragraph_number(self):
        neg = hn.neg_legal_ref(self.cases["legal_ref"])
        self.assertIn("§ 573", neg)  # 543 + 30

    def test_dates_numbers_changes_number(self):
        neg = hn.neg_dates_numbers(self.cases["dates_numbers"])
        self.assertIn("30", neg)  # 14 + 16

    def test_make_hard_negatives_collects_applicable_families(self):
        negs = hn.make_hard_negatives(self.cases["legal_ref"])
        self.assertIn("legal_ref", negs)
        # legal text also has numbers -> dates_numbers applies
        self.assertIn("dates_numbers", negs)

    def test_non_applicable_returns_none(self):
        self.assertIsNone(hn.neg_compound("Ein Satz ohne passendes Kompositum."))
        self.assertIsNone(hn.neg_regional_variant("Ein Satz ohne Monatsnamen."))


class TestFilters(unittest.TestCase):
    def test_accepts_good_pair(self):
        ok, reasons = hn.filter_pair(
            "Wie hoch ist die Mietkaution?",
            "Die Mietkaution darf höchstens drei Nettokaltmieten betragen.",
            negatives=["Die Maklerprovision beträgt zwei Nettokaltmieten."],
        )
        self.assertTrue(ok, reasons)

    def test_rejects_english(self):
        ok, reasons = hn.filter_pair("what is the capital", "the capital city is large")
        self.assertFalse(ok)
        self.assertIn("not_german", reasons)

    def test_rejects_query_equals_positive(self):
        ok, reasons = hn.filter_pair("Das ist ein Satz.", "das ist ein satz.")
        self.assertFalse(ok)
        self.assertIn("query_equals_positive", reasons)

    def test_rejects_negative_equal_to_positive(self):
        ok, reasons = hn.filter_pair(
            "Frage auf Deutsch?", "Die Antwort ist eindeutig und korrekt.",
            negatives=["die antwort ist eindeutig und korrekt."],
        )
        self.assertFalse(ok)
        self.assertIn("negative_equals_positive", reasons)


class TestPromptSpec(unittest.TestCase):
    def test_prompt_specs_parse_and_have_required_keys(self):
        spec = json.loads(
            (ROOT / "data" / "synthetic" / "prompt_specs.json").read_text(encoding="utf-8")
        )
        for key in ("version", "templates", "filters", "neg_types"):
            self.assertIn(key, spec)
        self.assertIn("query_from_passage", spec["templates"])


if __name__ == "__main__":
    unittest.main()
