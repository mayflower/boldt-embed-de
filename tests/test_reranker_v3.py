"""Tests for the v3 reranker: high-precision labels, uncertain-as-listwise-only, source-balance,
and a neutral-or-better promotion gate that catches v2's small GermanQuAD degradation."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import reranker_modern as RM  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
GATE = ROOT / "scripts" / "check_reranker_promotion_gate.py"
BUILD = ROOT / "scripts" / "build_reranker_candidates_v3.py"
TRAIN = ROOT / "scripts" / "train_modern_reranker.py"


def _v3_rows():
    return [json.loads(l) for l in (FIX / "reranker_v3_tiny.jsonl").read_text("utf-8").splitlines()]


class TestLabeling(unittest.TestCase):
    def test_v3_label_thresholds(self):
        self.assertEqual(RM.v3_label(5.0), 1)       # high-precision positive
        self.assertEqual(RM.v3_label(4.0), 1)
        self.assertEqual(RM.v3_label(1.0), 0)       # clear negative (<= 4 - 2)
        self.assertIsNone(RM.v3_label(3.0))         # uncertain band -> null
        self.assertIsNone(RM.v3_label(None))

    def test_uncertain_not_used_as_bce(self):
        pw = RM.candidate_lists_to_pointwise(_v3_rows())
        # 2 queries x (1 pos + 1 neg) labeled = 4; the two uncertain (label=null) are excluded
        self.assertEqual(len(pw), 4)
        docs = {e["document"] for e in pw}
        self.assertNotIn("Halbrelevant 3", docs)
        self.assertNotIn("Halbrelevant 6", docs)
        self.assertEqual({e["label"] for e in pw}, {0.0, 1.0})

    def test_listwise_includes_uncertain(self):
        lw = RM.candidate_lists_to_listwise(_v3_rows())
        # listwise keeps the FULL candidate list (incl. the uncertain doc) for the soft target
        self.assertEqual(len(lw[0]["documents"]), 3)

    def test_pairwise_strong_margin_only(self):
        rows = [{"query": "q", "candidates": [
            {"document": "p", "label": 1, "teacher_score": 5.0},
            {"document": "n_far", "label": 0, "teacher_score": 1.0},   # margin 4.0 -> kept
            {"document": "n_near", "label": 0, "teacher_score": 4.5},  # margin 0.5 -> dropped
        ]}]
        pairs = RM.candidate_lists_to_pairwise(rows, min_teacher_margin=2.0)
        negs = {p["negative"] for p in pairs}
        self.assertIn("n_far", negs)
        self.assertNotIn("n_near", negs)

    def test_training_summary(self):
        s = RM.reranker_training_summary(_v3_rows())
        self.assertEqual(s["uncertain"], 2)
        self.assertTrue(s["high_precision_positives"])
        self.assertGreaterEqual(s["min_positive_teacher_score"], 4.0)
        self.assertIn("bm25", s["candidate_source_distribution"])
        self.assertEqual(s["synthetic_share"], 0.0)
        self.assertIn("faq_real", s["separation_by_domain"])


class TestGate(unittest.TestCase):
    def _gate(self, dt, gq, *extra):
        return subprocess.run([sys.executable, str(GATE), "--dt-test", str(dt),
                               "--germanquad", str(gq), *extra], capture_output=True, text=True)

    def _lift(self, tmp, name, fs, rr):
        p = pathlib.Path(tmp) / f"{name}.json"
        p.write_text(json.dumps({"first_stage_ndcg@10": fs, "student_reranker_ndcg@10": rr}), "utf-8")
        return p

    def test_passes_on_neutral_and_lift(self):
        out = self._gate(FIX / "reranker_lift_dt_pass.json", FIX / "reranker_lift_germanquad_neutral.json")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)

    def test_fails_on_tiny_germanquad_degradation(self):
        with tempfile.TemporaryDirectory() as d:
            dt = self._lift(d, "dt", 0.95, 0.99)
            gq = self._lift(d, "gq", 0.886, 0.885)        # delta -0.001
            out = self._gate(dt, gq)
            self.assertEqual(out.returncode, 1)            # hard floor of 0 -> fail

    def test_catastrophic_degradation_fails(self):
        with tempfile.TemporaryDirectory() as d:
            dt = self._lift(d, "dt", 0.95, 0.90)          # delta -0.05 -> catastrophic
            gq = self._lift(d, "gq", 0.886, 0.886)
            out = self._gate(dt, gq)
            self.assertEqual(out.returncode, 1)
            self.assertIn("not_catastrophic", out.stdout)

    def test_low_precision_positives_block_and_override(self):
        with tempfile.TemporaryDirectory() as d:
            dt = self._lift(d, "dt", 0.95, 0.99)
            gq = self._lift(d, "gq", 0.886, 0.89)
            ts = pathlib.Path(d) / "ts.json"
            ts.write_text(json.dumps({"high_precision_positives": False, "positive_threshold": 2.0}), "utf-8")
            blocked = self._gate(dt, gq, "--training-summary", str(ts))
            self.assertEqual(blocked.returncode, 1)
            overridden = self._gate(dt, gq, "--training-summary", str(ts),
                                    "--allow-low-precision-positives")
            self.assertEqual(overridden.returncode, 0, overridden.stdout + overridden.stderr)


class TestCliDryRun(unittest.TestCase):
    def test_build_v3_dry_run_no_ml(self):
        out = subprocess.run(
            [sys.executable, str(BUILD), "--teacher-cache", str(FIX / "teacher_cache_calibrated.jsonl"),
             "--bm25-results", str(FIX / "bm25_results.jsonl"), "--output", "/tmp/_rv3.jsonl",
             "--dry-run"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("dry-run-ok", out.stdout)

    def test_train_v3_dry_run_no_ml(self):
        out = subprocess.run(
            [sys.executable, str(TRAIN), "--candidate-lists", str(FIX / "reranker_v3_tiny.jsonl"),
             "--loss", "mixed", "--output", "/tmp/_rv3model", "--pairwise-min-teacher-margin", "2.0",
             "--dry-run"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("dry-run-ok", out.stdout)
        self.assertIn("training_summary", out.stdout)


if __name__ == "__main__":
    unittest.main()
