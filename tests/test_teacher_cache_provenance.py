"""Regression tests for teacher-cache LICENSE PROVENANCE (the v2 by_license={"unknown":N} bug).

Pure stdlib (teacher/source_manifest are torch-free at import; verified in a subprocess too).
Covers: manifest license propagates to the cache summary; synthetic-inherits-source preserves
inherited metadata; unknown license fails; a disallowed (allowed_for_training=false) source fails.
"""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from boldt_embed import source_manifest as sm  # noqa: E402
from boldt_embed import teacher as T  # noqa: E402

MANIFEST = ROOT / "configs" / "data_sources_v2.json"
FIXTURE = ROOT / "tests" / "fixtures" / "teacher_cache_with_licenses.jsonl"
SUMMARIZER = ROOT / "scripts" / "summarize_teacher_cache.py"


def _entries():
    return {e.source_id: e for e in sm.load_source_manifest(MANIFEST)}


def _candidate(entry, query, document, domain=None, row=None):
    prov = sm.candidate_provenance(entry, row)
    return {"query_id": "q" + query, "doc_id": "d" + document, "query": query,
            "document": document, "positive": True, "domain": domain or entry.domain, **prov}


def _write_jsonl(rows):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    f.close()
    return f.name


class TestProvenancePropagation(unittest.TestCase):
    def test_manifest_license_propagates_to_summary(self):
        ents = _entries()
        cand = _candidate(ents["dt_de_dpr"], "frage", "dokument")
        self.assertEqual(cand["license"], "CC-BY-SA-4.0")
        self.assertEqual(cand["license_origin"], "manifest")
        row = T.make_cache_row(cand, embedding_score=0.8, reranker_score=7.0)
        # the cache row must carry the license through (the v2 bug dropped it here)
        self.assertEqual(row["license"], "CC-BY-SA-4.0")
        self.assertTrue(row["allowed_for_training"])
        summary = T.summarize_cache([row])
        self.assertIn("CC-BY-SA-4.0", summary["by_license"])
        self.assertEqual(summary["by_license"].get("unknown"), None)
        self.assertEqual(summary["unknown_license_rows"], 0)
        self.assertEqual(summary["by_license_origin"].get("manifest"), 1)

    def test_synthetic_inherits_source_preserves_inherited_metadata(self):
        ents = _entries()
        # synthetic source: manifest license is the inherits-marker; the raw row carries the
        # concrete inherited license + its seed source id.
        raw_row = {"license": "CC-BY-SA-4.0", "metadata": {"source_passage_source": "dt_de_dpr"}}
        cand = _candidate(ents["synthetic_faq_v2"], "faq frage", "passage", row=raw_row)
        self.assertEqual(cand["license_origin"], "inherited")
        self.assertEqual(cand["license"], "CC-BY-SA-4.0")        # concrete, not the marker
        self.assertEqual(cand["inherited_from_source_id"], "dt_de_dpr")
        row = T.make_cache_row(cand, reranker_score=2.5)
        summary = T.summarize_cache([row])
        self.assertEqual(summary["unknown_license_rows"], 0)
        self.assertEqual(summary["synthetic_inherits_source"]["rows"], 1)
        self.assertEqual(
            summary["synthetic_inherits_source"]["by_inherited_from_source_id"].get("dt_de_dpr"), 1)
        self.assertEqual(summary["by_license_origin"].get("inherited"), 1)

    def test_unknown_license_is_counted_and_fails_cli(self):
        # a row with NO license -> unknown_license_rows > 0
        bad = {"query_id": "q", "doc_id": "d", "query": "q", "document": "d", "positive": True,
               "source": "mystery", "domain": "web", "reranker_score": 5.0}
        summary = T.summarize_cache([bad])
        self.assertEqual(summary["unknown_license_rows"], 1)
        path = _write_jsonl([bad])
        out = subprocess.run([sys.executable, str(SUMMARIZER), "--input", path,
                              "--fail-on-unknown-license"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertIn("unknown", (out.stdout + out.stderr).lower())

    def test_disallowed_training_source_fails_cli(self):
        ents = _entries()
        # a public benchmark source has allowed_for_training=false -> must be flagged/fail
        cand = _candidate(ents["germanquad"], "frage", "kontext")
        self.assertFalse(cand["allowed_for_training"])
        row = T.make_cache_row(cand, reranker_score=8.0)
        summary = T.summarize_cache([row])
        self.assertEqual(summary["disallowed_for_training_rows"], 1)
        path = _write_jsonl([row])
        out = subprocess.run([sys.executable, str(SUMMARIZER), "--input", path,
                              "--fail-on-disallowed-training-source"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)


class TestShippedFixture(unittest.TestCase):
    def test_fixture_is_license_clean(self):
        rows = T.read_teacher_cache_jsonl(FIXTURE)
        summary = T.summarize_cache(rows)
        self.assertEqual(summary["unknown_license_rows"], 0)
        self.assertEqual(summary["disallowed_for_training_rows"], 0)
        self.assertNotIn("unknown", summary["by_license"])

    def test_summarizer_cli_passes_on_clean_fixture(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            outp = f.name
        out = subprocess.run([sys.executable, str(SUMMARIZER), "--input", str(FIXTURE),
                              "--output", outp, "--fail-on-unknown-license"],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertNotIn("torch", out.stdout.lower())  # CLI is stdlib-only on this path

    def test_summarizer_cli_is_stdlib_only(self):
        code = ("import sys; sys.path.insert(0, %r);"
                "import summarize_teacher_cache;"
                "assert 'torch' not in sys.modules; print('clean')") % str(ROOT / "scripts")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


class TestReleaseGateLicenseCheck(unittest.TestCase):
    def test_unknown_license_summary_flagged(self):
        import validate_release_2026 as VR
        with tempfile.TemporaryDirectory() as d:
            rd = pathlib.Path(d)
            (rd / "teacher-cache").mkdir()
            (rd / "teacher-cache" / "x.summary.json").write_text(
                json.dumps({"by_license": {"unknown": 7}, "unknown_license_rows": 7,
                            "disallowed_for_training_rows": 0}), encoding="utf-8")
            issues = VR.check_teacher_cache_license(rd)
            self.assertTrue(any("unknown_license" in i[0] for i in issues), issues)

    def test_clean_summary_passes(self):
        import validate_release_2026 as VR
        with tempfile.TemporaryDirectory() as d:
            rd = pathlib.Path(d)
            (rd / "teacher-cache").mkdir()
            (rd / "teacher-cache" / "x.summary.json").write_text(
                json.dumps({"by_license": {"CC-BY-4.0": 10}, "unknown_license_rows": 0,
                            "disallowed_for_training_rows": 0}), encoding="utf-8")
            self.assertEqual(VR.check_teacher_cache_license(rd), [])

    def test_old_schema_unknown_flagged(self):
        # historical summary with by_license unknown but no explicit field -> still caught
        import validate_release_2026 as VR
        with tempfile.TemporaryDirectory() as d:
            rd = pathlib.Path(d)
            (rd / "teacher-cache").mkdir()
            (rd / "teacher-cache" / "old.summary.json").write_text(
                json.dumps({"by_license": {"unknown": 44336}}), encoding="utf-8")
            issues = VR.check_teacher_cache_license(rd)
            self.assertTrue(any("unknown_license" in i[0] for i in issues), issues)


if __name__ == "__main__":
    unittest.main()
