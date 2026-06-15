"""Stdlib tests for v5 teacher-validated question generation. No ML, no network, no API calls."""
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import v5_question_generation as G  # noqa: E402


def passage(pid="p1", *, domain="web_nonfaq", license="CC-BY-4.0",
            title="Mietkaution", **kw):
    p = {"source_passage_id": pid, "domain": domain, "license": license, "title": title,
         "document": "Die Mietkaution darf hoechstens drei Nettokaltmieten betragen und ist "
                     "nach Auszug zurueckzuzahlen.", "source_id": f"src-{pid}"}
    p.update(kw)
    return p


class TestPrompts(unittest.TestCase):
    def test_prompt_export_deterministic(self):
        ps = [passage("a"), passage("b")]
        e1 = G.export_prompts(ps, G.QUERY_STYLES)
        e2 = G.export_prompts(ps, G.QUERY_STYLES)
        self.assertEqual(e1, e2)
        self.assertEqual(len(e1), 2 * len(G.QUERY_STYLES))

    def test_prompt_is_german_json_passage_answerable(self):
        pr = G.build_prompt(passage(), "definition")
        self.assertIn("AUSSCHLIESSLICH", pr)               # answerable only from passage
        self.assertIn("JSON", pr)                          # requires JSON output
        self.assertIn("Textabschnitt", pr)                 # German instruction
        self.assertIn("definition", pr)                    # the requested style
        self.assertIn("answerable_without_passage", pr)    # reject-flag instruction

    def test_build_prompt_unknown_style_raises(self):
        with self.assertRaises(ValueError):
            G.build_prompt(passage(), "not_a_style")


class TestGenerationAndValidation(unittest.TestCase):
    def test_templates_cover_all_styles_and_validate(self):
        rows = G.generate_from_templates([passage()], G.QUERY_STYLES)
        styles = {r["query_style"] for r in rows}
        self.assertEqual(styles, set(G.QUERY_STYLES))      # full style coverage
        for i, r in enumerate(rows):
            self.assertEqual(G.validate_generated_row(r, i), [])
            self.assertTrue(r["synthetic_query"])
            self.assertTrue(r["must_teacher_validate"])

    def test_generated_jsonl_validation_catches_bad_rows(self):
        good = G.generate_from_templates([passage()], ("definition",))[0]
        self.assertEqual(G.validate_generated_row(good, 0), [])

        bad_synth = dict(good); bad_synth["synthetic_query"] = False
        self.assertTrue(any("synthetic_query" in e for e in G.validate_generated_row(bad_synth, 0)))

        bad_val = dict(good); bad_val["must_teacher_validate"] = False
        self.assertTrue(any("must_teacher_validate" in e for e in G.validate_generated_row(bad_val, 0)))

        bad_style = dict(good); bad_style["query_style"] = "nope"
        self.assertTrue(any("query_style" in e for e in G.validate_generated_row(bad_style, 0)))

        no_query = dict(good); no_query["query"] = ""
        self.assertTrue(any("'query'" in e for e in G.validate_generated_row(no_query, 0)))

    def test_unknown_license_fails(self):
        # passage-level
        self.assertTrue(any("license" in e
                            for e in G.validate_passage(passage(license="proprietary-internal"), 0)))
        # row-level
        row = G.make_row(passage(license="proprietary-internal"), "Was?", "definition",
                         "dry_run_templates")
        self.assertTrue(any("license" in e for e in G.validate_generated_row(row, 0)))

    def test_leakage_passage_rejected(self):
        self.assertTrue(G.validate_passage(passage(public_benchmark=True), 0))
        self.assertTrue(G.validate_passage(passage("germanquad-test-7"), 0))


class TestLocalLlmJoin(unittest.TestCase):
    def test_join_and_reject_answerable_without_passage(self):
        ps = {"p1": passage("p1")}
        llm = [
            {"source_passage_id": "p1", "query": "Wie hoch darf die Kaution sein?",
             "query_style": "germanquad_fact", "answerable_without_passage": False},
            {"source_passage_id": "p1", "query": "Was ist die Hauptstadt von Frankreich?",
             "query_style": "definition", "answerable_without_passage": True},  # rejected
        ]
        kept, rejected, errors = G.rows_from_local_llm(llm, ps)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(errors, [])
        self.assertEqual(kept[0]["license"], "CC-BY-4.0")    # provenance from trusted passage

    def test_missing_passage_and_bad_style_error(self):
        ps = {"p1": passage("p1")}
        llm = [{"source_passage_id": "ghost", "query": "x", "query_style": "definition"},
               {"source_passage_id": "p1", "query": "y", "query_style": "bogus"}]
        kept, rejected, errors = G.rows_from_local_llm(llm, ps)
        self.assertEqual(kept, [])
        self.assertEqual(len(errors), 2)


class TestProvisional(unittest.TestCase):
    def test_generated_rows_not_training_ready_until_teacher_score(self):
        row = G.generate_from_templates([passage()], ("definition",))[0]
        self.assertFalse(G.is_training_ready(row, 4.0))      # provisional, no teacher_score
        scored = dict(row); scored["teacher_score"] = 4.5
        self.assertTrue(G.is_training_ready(scored, 4.0))
        low = dict(row); low["teacher_score"] = 3.0
        self.assertFalse(G.is_training_ready(low, 4.0))

    def test_summary_proves_coverage_and_provisional(self):
        rows = G.generate_from_templates([passage("a"), passage("b")], G.QUERY_STYLES)
        s = G.summarize(rows, mode="dry_run_templates")
        self.assertEqual(s["query_styles_missing"], [])
        self.assertTrue(s["all_must_teacher_validate"])
        self.assertEqual(s["training_ready_rows"], 0)


class TestNoMlImports(unittest.TestCase):
    def test_template_generation_imports_no_torch_in_subprocess(self):
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "from boldt_embed import v5_question_generation as G\n"
            "rows = G.generate_from_templates([{'source_passage_id':'p','domain':'web_nonfaq',"
            "'license':'CC-BY-4.0','document':'Ein deutscher Textabschnitt.','title':'T'}], "
            "G.QUERY_STYLES)\n"
            "assert len(rows) == len(G.QUERY_STYLES)\n"
            "assert 'torch' not in sys.modules, 'torch must not be imported'\n"
            "print('OK')\n" % str(ROOT / "src")
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("OK", r.stdout)


if __name__ == "__main__":
    unittest.main()
