"""Stdlib tests for the teacher-cache layer + build_teacher_cache.py dry-run.

No ML deps, no network, no model downloads. The dry-run no-ML guarantee is verified by
importing the script module in a subprocess and asserting torch / sentence_transformers
never entered sys.modules.
"""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import teacher as T  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "teacher_candidates.jsonl"
SCRIPT = ROOT / "scripts" / "build_teacher_cache.py"
TEACHER_CFG = ROOT / "configs" / "teacher_models.json"


class TestSchema(unittest.TestCase):
    def test_valid_candidate_passes(self):
        rows = T.read_candidates(FIXTURE)
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertEqual(T.validate_candidate_record(r), [])

    def test_missing_field_flagged(self):
        errs = T.validate_candidate_record({"query_id": "q", "doc_id": "d", "query": "x"})
        self.assertTrue(any("document" in e for e in errs), errs)

    def test_bad_positive_type_flagged(self):
        errs = T.validate_candidate_record(
            {"query_id": "q", "doc_id": "d", "query": "x", "document": "y", "positive": "yes"})
        self.assertTrue(any("positive" in e for e in errs), errs)

    def test_make_cache_row_has_all_fields(self):
        cand = T.read_candidates(FIXTURE)[0]
        row = T.make_cache_row(cand, embedding_teacher_model="m", embedding_score=0.9,
                               created_at="2026-01-01T00:00:00+00:00")
        for key in T.CACHE_FIELDS:
            self.assertIn(key, row)
        self.assertEqual(row["score_version"], T.SCORE_VERSION)
        self.assertEqual(row["embedding_score"], 0.9)
        self.assertIsNone(row["reranker_score"])


class TestCacheIO(unittest.TestCase):
    def test_write_then_read_roundtrip(self):
        cands = T.read_candidates(FIXTURE)
        rows = [T.make_cache_row(c, embedding_teacher_model="m", embedding_score=0.5,
                                 created_at="t") for c in cands]
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "sub" / "cache.jsonl"  # parent auto-created
            n = T.write_teacher_cache_jsonl(out, rows)
            self.assertEqual(n, 3)
            back = T.read_teacher_cache_jsonl(out)
            self.assertEqual([r["doc_id"] for r in back], ["d1", "d2", "d3"])

    def test_read_missing_file_is_empty(self):
        self.assertEqual(T.read_teacher_cache_jsonl("/no/such/file.jsonl"), [])


class TestResume(unittest.TestCase):
    def test_existing_keys_and_filter(self):
        cands = T.read_candidates(FIXTURE)
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "cache.jsonl"
            # Pretend q1/d1 was already scored in a prior run.
            T.write_teacher_cache_jsonl(out, [T.make_cache_row(cands[0], created_at="t")])
            done = T.existing_cache_keys(out)
            self.assertIn(("q1", "d1"), done)
            remaining = T.filter_unscored(cands, done)
            self.assertEqual([T.cache_key(c) for c in remaining], [("q1", "d2"), ("q2", "d3")])


class TestDryRunNoML(unittest.TestCase):
    def test_dry_run_does_not_import_ml(self):
        code = (
            "import sys; sys.argv=['build_teacher_cache.py','--input',%r,"
            "'--teacher-config',%r,'--mode','both','--dry-run'];"
            "sys.path.insert(0, %r);"
            "import build_teacher_cache as b; rc=b.main();"
            "assert rc==0, rc;"
            "assert 'torch' not in sys.modules, 'torch imported';"
            "assert 'sentence_transformers' not in sys.modules, 'st imported';"
            "print('PLANNED_OK')"
        ) % (str(FIXTURE), str(TEACHER_CFG), str(ROOT / "scripts"))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("PLANNED_OK", out.stdout)
        self.assertIn("dry-run-ok", out.stdout)

    def test_dry_run_cli_prints_planned_rows(self):
        out = subprocess.run(
            [sys.executable, str(SCRIPT), "--input", str(FIXTURE),
             "--teacher-config", str(TEACHER_CFG), "--mode", "embedding", "--dry-run"],
            capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("DRY RUN", out.stdout)
        # The first planned row should be valid JSON carrying the embedding teacher name.
        planned = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
        self.assertTrue(planned)
        row = json.loads(planned[0])
        self.assertEqual(row["embedding_teacher_model"], "Qwen/Qwen3-Embedding-8B")
        self.assertIsNone(row["reranker_teacher_model"])  # mode=embedding only


if __name__ == "__main__":
    unittest.main()
