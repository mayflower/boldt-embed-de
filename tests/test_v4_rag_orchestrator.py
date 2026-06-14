"""Tests for the v4 RAG reranker orchestrator: safe-by-default, deterministic plan, RAG-focused."""
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v4_rag_reranker_experiment.py"
LOCAL_RAG = ROOT / "data" / "eval" / "rag_local"


def _run(work, *extra):
    return subprocess.run([sys.executable, str(SCRIPT), "--work-dir", str(work), *extra],
                          capture_output=True, text=True, cwd=str(ROOT))


def _stage_names(work):
    return [s["name"] for s in json.loads((pathlib.Path(work) / "STATUS.json").read_text())["stages"]]


class TestDryRun(unittest.TestCase):
    def test_deterministic_command_list(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self.assertEqual(_run(a, "--mode", "dry-run").returncode, 0)
            self.assertEqual(_run(b, "--mode", "dry-run").returncode, 0)
            s1, s2 = _stage_names(a), _stage_names(b)
            self.assertEqual(s1, s2)
            self.assertEqual(s1[0], "build_webfaq_eval")
            self.assertEqual(s1[-1], "promotion_gate")
            for required in ("lift_webfaq", "lift_germanquad", "lift_dt_test", "train_rag_reranker"):
                self.assertIn(required, s1)

    def test_no_ml_and_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            r = _run(d, "--mode", "dry-run")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("dry-run-ok", r.stdout)
            for f in ("COMMANDS.md", "STATUS.json", "V4_RAG_RESULTS.json", "V4_RAG_RESULTS.md"):
                self.assertTrue((pathlib.Path(d) / f).exists(), f)
            self.assertEqual(json.loads((pathlib.Path(d) / "V4_RAG_RESULTS.json").read_text())["verdict"],
                             "planned")

    def test_imports_no_torch(self):
        code = ("import sys; sys.path.insert(0, %r); sys.path.insert(0, %r);"
                "import run_v4_rag_reranker_experiment;"
                "assert 'torch' not in sys.modules; print('clean')") % (
                    str(ROOT / "scripts"), str(ROOT / "src"))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


class TestSafety(unittest.TestCase):
    def test_full_blocked_without_flag(self):
        with tempfile.TemporaryDirectory() as d:
            r = _run(d, "--mode", "full")
            self.assertEqual(r.returncode, 2)
            self.assertIn("i-understand-this-runs-gpu", r.stderr)

    def test_eval_leakage_guard_wired(self):
        # the train stage must pass --eval-query-ids = WebFAQ held-out queries
        with tempfile.TemporaryDirectory() as d:
            _run(d, "--mode", "dry-run")
            cmds = (pathlib.Path(d) / "COMMANDS.md").read_text("utf-8")
            self.assertIn("--eval-query-ids", cmds)
            self.assertIn("eval/webfaq/queries.jsonl", cmds)


class TestOptionalSets(unittest.TestCase):
    def test_local_rag_optional(self):
        with tempfile.TemporaryDirectory() as d:
            _run(d, "--mode", "dry-run")
            names = _stage_names(d)
            present = LOCAL_RAG.exists()
            self.assertEqual(any("local_rag" in n for n in names), present)

    def test_local_rag_included_when_present(self):
        if LOCAL_RAG.exists():
            self.skipTest("local rag dir already exists")
        try:
            LOCAL_RAG.mkdir(parents=True)
            for f in ("corpus.jsonl", "queries.jsonl", "qrels.jsonl"):
                (LOCAL_RAG / f).write_text("", encoding="utf-8")
            with tempfile.TemporaryDirectory() as d:
                _run(d, "--mode", "dry-run")
                self.assertTrue(any("local_rag" in n for n in _stage_names(d)))
        finally:
            shutil.rmtree(LOCAL_RAG, ignore_errors=True)

    def test_gerdalir_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as d:
            # default: no gerdalir stage
            _run(d, "--mode", "dry-run")
            self.assertNotIn("lift_gerdalir", _stage_names(d))
        with tempfile.TemporaryDirectory() as d:
            _run(d, "--mode", "dry-run", "--with-gerdalir-diagnostic")
            self.assertIn("lift_gerdalir", _stage_names(d))
            # the gerdalir lift command is marked --diagnostic so the gate ignores it
            self.assertIn("--diagnostic", (pathlib.Path(d) / "COMMANDS.md").read_text("utf-8"))


if __name__ == "__main__":
    unittest.main()
