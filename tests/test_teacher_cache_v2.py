"""Stdlib tests for v2 teacher-cache sharding, summary, filtering. No ML, no network."""
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import teacher as T  # noqa: E402

CACHE = ROOT / "tests" / "fixtures" / "teacher_cache_small.jsonl"
CANDS = ROOT / "tests" / "fixtures" / "teacher_candidates.jsonl"
TEACHER_CFG = ROOT / "configs" / "teacher_models.json"
BUILD = ROOT / "scripts" / "build_teacher_cache.py"
SUMM = ROOT / "scripts" / "summarize_teacher_cache.py"


class TestSharding(unittest.TestCase):
    def test_shard_sizes_deterministic(self):
        rows = [{"i": i} for i in range(5)]
        a = T.shard_candidates(rows, 2)
        self.assertEqual([len(s) for s in a], [2, 2, 1])
        self.assertEqual(a, T.shard_candidates(rows, 2))

    def test_shard_path(self):
        self.assertTrue(T.shard_path("/x", "qwen3_v2", 3).endswith("qwen3_v2.shard-00003.jsonl"))


class TestSummary(unittest.TestCase):
    def setUp(self):
        self.rows = T.read_teacher_cache_jsonl(CACHE)

    def test_summary_fields(self):
        s = T.summarize_cache(self.rows)
        self.assertEqual(s["total_rows"], 5)
        self.assertEqual(s["positives"], 2)
        self.assertIn("syn", s["by_source"])
        self.assertEqual(s["reranker_score"]["n"], 5)
        self.assertIn("median", s["embedding_score"])
        self.assertTrue(isinstance(s["suspicious_low_positives"], list))


class TestFilter(unittest.TestCase):
    def setUp(self):
        self.rows = T.read_teacher_cache_jsonl(CACHE)

    def test_threshold_routes_low_positive_to_review(self):
        split = T.filter_cache(self.rows, reranker_threshold=5.0)  # d1 pos=4.1 -> review, d3=5.2 keep
        review_ids = [r["doc_id"] for r in split["review"]]
        self.assertIn("d1", review_ids)
        self.assertNotIn("d3", review_ids)
        # negatives always kept; reasons recorded
        self.assertTrue(all(r.get("filtering_reason") is None for r in split["kept"] if not r["positive"]))
        self.assertTrue(all("below_reranker_threshold" in r["filtering_reason"] for r in split["review"]))

    def test_low_threshold_keeps_all_positives(self):
        split = T.filter_cache(self.rows, reranker_threshold=0.0)
        self.assertEqual(split["review"], [])


class TestCLIs(unittest.TestCase):
    def test_build_sharded_dry_run_no_ml(self):
        out = subprocess.run(
            [sys.executable, str(BUILD), "--input", str(CANDS), "--teacher-config", str(TEACHER_CFG),
             "--output", "/tmp/qwen3_v2.jsonl", "--mode", "both", "--shard-size", "2", "--dry-run"],
            capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("[shard]", out.stdout)
        self.assertIn("dry-run-ok", out.stdout)
        self.assertIn("manifest", out.stdout)

    def test_summarize_cli(self):
        out = subprocess.run([sys.executable, str(SUMM), "--input", str(CACHE)],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn('"total_rows": 5', out.stdout)


if __name__ == "__main__":
    unittest.main()
