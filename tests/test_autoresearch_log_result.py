"""Tests for the AutoResearch results.tsv logger (stdlib)."""
import importlib.util
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


L = _load("ar_log_result")


def _metrics_doc():
    return {
        "run_id": "trial-1", "mode": "real", "status": "ok", "budget_minutes": 20,
        "elapsed_seconds": 1.2, "invalid_for_default_loop": False, "config_path": "configs/x.json",
        "git": {"commit": "abc123", "dirty": True},
        "metrics": {
            "webfaq": {"recall@100": 0.95, "ndcg@10": 0.66, "mrr@10": 0.55},
            "germanquad": {"ndcg@10": 0.88}, "dt_test": {"ndcg@10": 0.95},
            "matryoshka": {"retention_256": 0.96},
            "leakage": {"hits": 0, "status": "clean"},
            "system": {"vram_gb": 12.0, "throughput_pairs_per_sec": 900.0},
        },
    }


class BuildRowTests(unittest.TestCase):
    def test_maps_fields(self):
        row = L.build_row(_metrics_doc(), {"status": "pass", "score": 0.155}, None, "a note")
        self.assertEqual(row["run_id"], "trial-1")
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["score"], "0.155")
        self.assertEqual(row["webfaq_recall100"], "0.95")
        self.assertEqual(row["m256_retention"], "0.96")
        self.assertEqual(row["commit"], "abc123")
        self.assertEqual(row["mode"], "real")
        self.assertEqual(row["leakage_status"], "clean")
        self.assertEqual(row["notes"], "a note")

    def test_missing_score_falls_back_to_metrics_status(self):
        row = L.build_row(_metrics_doc(), None, None, None)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["score"], "")

    def test_status_override(self):
        row = L.build_row(_metrics_doc(), {"status": "pass", "score": 0.1}, "discard", None)
        self.assertEqual(row["status"], "discard")

    def test_notes_sanitized(self):
        row = L.build_row(_metrics_doc(), None, None, "line1\nline2\twith tab")
        self.assertNotIn("\n", row["notes"])
        self.assertNotIn("\t", row["notes"])


class AppendTests(unittest.TestCase):
    def test_header_then_append(self):
        with tempfile.TemporaryDirectory() as d:
            tsv = pathlib.Path(d) / "results.tsv"
            L.append_row(tsv, L.build_row(_metrics_doc(), None, None, "first"))
            L.append_row(tsv, L.build_row(_metrics_doc(), None, None, "second"))
            lines = tsv.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 3)                 # header + 2 rows
            self.assertEqual(lines[0].split("\t"), L.COLUMNS)
            self.assertEqual(lines[0], "\t".join(L.COLUMNS))   # header only once


if __name__ == "__main__":
    unittest.main()
