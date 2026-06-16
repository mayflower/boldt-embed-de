"""Tests for the dense-recall STOP gate (stdlib, no ML)."""
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_dense_recall_gate",
                                              ROOT / "scripts" / "check_dense_recall_gate.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


def _wf(**kw):
    base = {"recall_at_100": 0.96, "positive_in_top_50": 0.92, "oracle_ndcg_at_10": 0.96,
            "candidate_union_size": 200, "missing_positive_rate": 0.03}
    base.update(kw)
    return {"webfaq": base}


class GateLogicTests(unittest.TestCase):
    def test_passes_when_recall_sufficient(self):
        self.assertEqual(G.dense_recall_gate(_wf())["status"], "pass")

    def test_fails_low_recall_100_and_flags_positives_absent(self):
        g = G.dense_recall_gate(_wf(recall_at_100=0.70))
        self.assertEqual(g["status"], "fail")
        self.assertIn("webfaq_recall_at_100", [c["check"] for c in g["failing"]])
        self.assertTrue(g["positives_absent"])
        self.assertIn("missing positives", g["stop_reason"])

    def test_fails_low_top50(self):
        g = G.dense_recall_gate(_wf(positive_in_top_50=0.80))
        self.assertEqual(g["status"], "fail")
        self.assertIn("webfaq_positive_in_top_50", [c["check"] for c in g["failing"]])

    def test_fails_high_missing_rate_is_positives_absent(self):
        g = G.dense_recall_gate(_wf(missing_positive_rate=0.30))
        self.assertEqual(g["status"], "fail")
        self.assertTrue(g["positives_absent"])

    def test_fails_tiny_union(self):
        g = G.dense_recall_gate(_wf(candidate_union_size=5))
        self.assertEqual(g["status"], "fail")
        self.assertIn("candidate_union_size", [c["check"] for c in g["failing"]])

    def test_missing_webfaq_metrics_fails(self):
        g = G.dense_recall_gate({})
        self.assertEqual(g["status"], "fail")
        self.assertTrue(g["positives_absent"])

    def test_local_rag_gated_when_present(self):
        m = _wf(); m["local_rag"] = {"recall_at_100": 0.60}
        g = G.dense_recall_gate(m)
        self.assertEqual(g["status"], "fail")
        self.assertIn("local_rag_recall_at_100", [c["check"] for c in g["failing"]])

    def test_targets_are_tunable(self):
        # the same metrics pass under a relaxed top-50 target
        self.assertEqual(
            G.dense_recall_gate(_wf(positive_in_top_50=0.85),
                                targets={"positive_in_top_50": 0.80})["status"], "pass")


class ExtractionTests(unittest.TestCase):
    def test_extract_prefers_dense_recall_report(self):
        rr = {"dense_v6": {"recall@100": 0.964, "recall@50": 0.883}}
        ur = {"positive_present_rate": 0.966, "list_size": 200,
              "union_recall": {"recall@50": 0.808, "recall@200": 0.966}}
        d = pathlib.Path(tempfile.mkdtemp())
        (d / "rr.json").write_text(json.dumps(rr)); (d / "ur.json").write_text(json.dumps(ur))
        m = G.extract_webfaq_metrics(str(d / "rr.json"), str(d / "ur.json"))
        self.assertEqual(m["recall_at_100"], 0.964)
        self.assertEqual(m["positive_in_top_50"], 0.883)   # dense recall@50, preferred over union
        self.assertEqual(m["candidate_union_size"], 200)
        self.assertAlmostEqual(m["missing_positive_rate"], 0.034, places=3)
        self.assertEqual(m["oracle_ndcg_at_10"], 0.966)


class StopFileTests(unittest.TestCase):
    def _reports(self, d, recall100, top50):
        (d / "rr.json").write_text(json.dumps(
            {"dense_v6": {"recall@100": recall100, "recall@50": top50}}))
        (d / "ur.json").write_text(json.dumps(
            {"positive_present_rate": 0.97, "list_size": 200, "union_recall": {}}))

    def test_cli_writes_stop_file_on_fail_and_removes_on_pass(self):
        d = pathlib.Path(tempfile.mkdtemp())
        stop = d / "STOP_RERANKER_TRAINING.md"
        # fail: low recall
        self._reports(d, 0.70, 0.65)
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_dense_recall_gate.py"),
                            "--recall-report", str(d / "rr.json"), "--union-report", str(d / "ur.json"),
                            "--output", str(d / "gate.json"), "--stop-file", str(stop)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertTrue(stop.exists())
        self.assertIn("cannot recover missing positives", stop.read_text())
        # pass: high recall -> stop file removed
        self._reports(d, 0.97, 0.95)
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_dense_recall_gate.py"),
                            "--recall-report", str(d / "rr.json"), "--union-report", str(d / "ur.json"),
                            "--output", str(d / "gate.json"), "--stop-file", str(stop)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse(stop.exists())

    def test_quality_only_miss_is_advisory_no_stop_file(self):
        # recall@100 high (positives present) but top-50 below target -> fail, but NO blocking STOP
        d = pathlib.Path(tempfile.mkdtemp())
        stop = d / "STOP_RERANKER_TRAINING.md"
        self._reports(d, 0.964, 0.85)   # recall@100 0.964 (present), top50 0.85 < 0.90
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_dense_recall_gate.py"),
                            "--recall-report", str(d / "rr.json"), "--union-report", str(d / "ur.json"),
                            "--output", str(d / "gate.json"), "--stop-file", str(stop)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 1, r.stderr)      # targets missed -> fail
        self.assertIn("ADVISORY", r.stdout)
        self.assertFalse(stop.exists())                  # but positives present -> no blocking stop
        g = json.loads((d / "gate.json").read_text())
        self.assertFalse(g["positives_absent"])


class TrainerRefusalTests(unittest.TestCase):
    def test_trainer_refuses_when_stop_active(self):
        # the real STOP file at repo root governs the real trainer; here we just verify the trainer
        # script refuses (exit 2) when a STOP file is present and --force-research-run is absent.
        # Use a tiny candidate-lists file; the refusal must happen before any training.
        d = pathlib.Path(tempfile.mkdtemp())
        cl = d / "lists.jsonl"
        cl.write_text(json.dumps({"query_id": "q1", "query": "q", "positive_doc_ids": ["g"],
                                  "domain": "faq_real",
                                  "candidates": [{"doc_id": "g", "text": "t", "label": 1,
                                                  "high_precision_positive": True,
                                                  "teacher_score": 8.0, "first_stage_rank": 0}]}))
        real_stop = ROOT / "STOP_RERANKER_TRAINING.md"
        created = False
        if not real_stop.exists():
            real_stop.write_text("# test stop\n"); created = True
        try:
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_v6_raw_rag_reranker.py"),
                                "--candidate-lists", str(cl), "--output", str(d / "ckpt"),
                                "--report", str(d / "rep.json"), "--dry-run"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertIn("REFUSING to train", r.stderr)
        finally:
            if created:
                real_stop.unlink()


if __name__ == "__main__":
    unittest.main()
