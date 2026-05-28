import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import instructions  # noqa: E402


class TestInstructions(unittest.TestCase):
    def test_query_template_substitution(self):
        tmpl = "Instruct: Repräsentiere die Suchanfrage.\nQuery: {query}"
        out = instructions.format_query(tmpl, "kündigungsfrist mietvertrag")
        self.assertIn("kündigungsfrist mietvertrag", out)
        self.assertIn("Instruct:", out)
        self.assertNotIn("{query}", out)

    def test_document_template_passthrough(self):
        self.assertEqual(instructions.format_document("{document}", "Ein Dokument."), "Ein Dokument.")

    def test_empty_template_returns_text(self):
        self.assertEqual(instructions.format_query("", "abc"), "abc")
        self.assertEqual(instructions.format_document("", "abc"), "abc")

    def test_prefix_template_without_placeholder(self):
        self.assertEqual(instructions.format_query("passage: ", "x"), "passage: x")


if __name__ == "__main__":
    unittest.main()
