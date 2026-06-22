"""Smoke test for the CLI loop orchestrator (dry-run path; stdlib only)."""
import contextlib
import importlib.util
import io
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LOOP = _load("ar_loop")


def _real_baseline():
    return {"status": "ok", "mode": "real", "metrics": {
        "webfaq": {"recall@100": 0.90, "ndcg@10": 0.60, "mrr@10": 0.50},
        "germanquad": {"ndcg@10": 0.88}, "dt_test": {"ndcg@10": 0.95},
        "matryoshka": {"retention_256": 0.96}, "leakage": {"hits": 0, "status": "clean"},
        "system": {"vram_gb": 0.0, "throughput_pairs_per_sec": 1.0}}}


class LoopDryRunTests(unittest.TestCase):
    def test_one_iteration_runs_and_is_not_promotable(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            baseline = d / "baseline.json"
            baseline.write_text(json.dumps(_real_baseline()), encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = LOOP.main(["--dry-run", "--run-id", "t1", "--out-root", str(d / "runs"),
                                "--results", str(d / "results.tsv"), "--baseline", str(baseline)])
            verdict = json.loads(buf.getvalue())
            # the loop ran end-to-end ...
            self.assertEqual(verdict["trial_status"], "ok")
            self.assertEqual(verdict["mode"], "dry_run")
            self.assertTrue((d / "runs" / "t1" / "metrics.json").exists())
            self.assertTrue((d / "results.tsv").exists())
            # ... but a dry-run can never be promotable (scorer rejects non-real runs)
            self.assertEqual(rc, 1)
            self.assertFalse(verdict["promotable"])
            self.assertEqual(verdict["score_status"], "fail")
            self.assertIn("not_a_real_run", verdict["failed_gates"])

    def test_real_and_dry_run_conflict(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(LOOP.main(["--dry-run", "--real"]), 2)


if __name__ == "__main__":
    unittest.main()
