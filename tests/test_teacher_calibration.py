"""Tests for domain-aware teacher-threshold calibration (pure stdlib).

Core guarantee: the reranker positive set is a higher-precision, strict subset of the embedder
set — they no longer share one noisy threshold (the v2 mistake)."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import teacher_calibration as tc  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
CACHE = FIX / "teacher_cache_calibration.jsonl"
CFG = FIX / "v3_real_domain_generalization.json"
SCRIPT = ROOT / "scripts" / "calibrate_teacher_thresholds.py"


def _cache():
    return [json.loads(l) for l in CACHE.read_text("utf-8").splitlines() if l.strip()]


def _gates():
    return json.loads(CFG.read_text("utf-8")).get("domain_quality_gates", {}).get("min_real_domain_accepted")


def _row(dom, score, positive=True, lic="CC-BY-4.0"):
    return {"query_id": f"{dom}{score}{positive}", "doc_id": "d", "query": "q", "document": "doc",
            "domain": dom, "source": f"{dom}_src", "license": lic, "license_origin": "manifest",
            "positive": positive, "reranker_score": score, "embedding_score": 0.5}


class TestSweep(unittest.TestCase):
    def test_threshold_sweep_keys_and_monotonic(self):
        sweep = tc.acceptance_by_threshold(_cache())["by_threshold"]
        self.assertEqual(list(sweep.keys()), ["-2", "0", "1", "2", "3", "4", "5"])
        counts = [sweep[k]["accepted"] for k in sweep]
        self.assertEqual(counts, sorted(counts, reverse=True))  # non-increasing as threshold rises

    def test_sweep_by_domain(self):
        bydom = tc.acceptance_by_threshold_grouped(_cache(), "domain")
        self.assertIn("faq_real", bydom)
        self.assertGreaterEqual(bydom["faq_real"]["by_threshold"]["2"]["accepted"],
                                bydom["faq_real"]["by_threshold"]["4"]["accepted"])


class TestCalibrate(unittest.TestCase):
    def test_reranker_is_stricter_higher_precision(self):
        rep = tc.calibrate(_cache(), embedder_threshold=2.0, reranker_threshold=4.0,
                           min_real_domain_accepted=_gates())
        self.assertEqual(rep["status"], "pass", rep["failing_gates"])
        self.assertLess(rep["reranker_accepted"], rep["embedder_accepted"])
        # higher precision: every reranker-kept positive scores >= 4.0; embedder set has some < 4.0
        emb = rep.pop("_embedder_kept"); rr = rep.pop("_reranker_kept")
        self.assertTrue(all(r["reranker_score"] >= 4.0 for r in rr))
        self.assertTrue(any(r["reranker_score"] < 4.0 for r in emb))

    def test_per_domain_threshold_override(self):
        rep = tc.calibrate(_cache(), reranker_threshold=4.0,
                           per_domain_reranker={"faq_real": 5.5}, min_real_domain_accepted=_gates())
        # faq_real strong positives are 5.0 < 5.5 -> none pass the per-domain reranker threshold
        self.assertEqual(rep["reranker_accepted_by_domain"].get("faq_real", 0), 0)
        self.assertGreater(rep["reranker_accepted_by_domain"].get("web", 0), 0)  # web unaffected

    def test_low_score_positive_reported(self):
        rep = tc.calibrate(_cache(), embedder_threshold=2.0)
        scores = [e["reranker_score"] for e in rep["low_score_positives"]]
        self.assertIn(0.5, scores)

    def test_high_score_rejected_reported(self):
        rep = tc.calibrate(_cache(), reranker_threshold=4.0)
        self.assertTrue(rep["high_score_rejected"])
        self.assertTrue(all(e["reranker_score"] >= 4.0 for e in rep["high_score_rejected"]))

    def test_unknown_license_fails(self):
        rows = [_row("faq_real", 5.0, lic="unknown") for _ in range(3)]
        rep = tc.calibrate(rows)
        self.assertEqual(rep["status"], "fail")
        self.assertTrue(any(g["gate"] == "license_unknown_rows_zero" for g in rep["failing_gates"]))

    def test_real_domain_min_fails(self):
        rows = [_row("faq_real", 5.0)]   # only 1 accepted, floor 100
        rep = tc.calibrate(rows, min_real_domain_accepted={"faq_real": 100})
        self.assertEqual(rep["status"], "fail")
        self.assertTrue(any(g["gate"] == "real_domain_min_accepted" and g["domain"] == "faq_real"
                            for g in rep["failing_gates"]))

    def test_suspicious_rate_fails(self):
        rows = [_row("web", 0.1) for _ in range(8)] + [_row("web", 5.0) for _ in range(2)]
        rep = tc.calibrate(rows, embedder_threshold=2.0, max_suspicious_rate=0.3)
        self.assertEqual(rep["status"], "fail")
        self.assertTrue(any(g["gate"] == "suspicious_positive_rate" for g in rep["failing_gates"]))


class TestCli(unittest.TestCase):
    def test_listed_command_passes_and_splits(self):
        with tempfile.TemporaryDirectory() as d:
            j, m, e, r = (pathlib.Path(d) / x for x in ("c.json", "c.md", "e.jsonl", "r.jsonl"))
            out = subprocess.run(
                [sys.executable, str(SCRIPT), "--teacher-cache", str(CACHE), "--config", str(CFG),
                 "--output", str(j), "--markdown", str(m),
                 "--embedder-output", str(e), "--reranker-output", str(r)],
                capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            n_emb = len(e.read_text("utf-8").splitlines())
            n_rr = len(r.read_text("utf-8").splitlines())
            self.assertGreater(n_emb, n_rr)          # reranker set is a strict subset
            self.assertIn("calibration", m.read_text("utf-8").lower())

    def test_cli_blocks_on_unknown_license(self):
        with tempfile.TemporaryDirectory() as d:
            bad = pathlib.Path(d) / "bad.jsonl"
            bad.write_text("\n".join(json.dumps(_row("faq_real", 5.0, lic="unknown"))
                                     for _ in range(3)) + "\n", encoding="utf-8")
            out = subprocess.run(
                [sys.executable, str(SCRIPT), "--teacher-cache", str(bad),
                 "--output", str(pathlib.Path(d) / "c.json"),
                 "--markdown", str(pathlib.Path(d) / "c.md")], capture_output=True, text=True)
            self.assertEqual(out.returncode, 1)

    def test_module_is_stdlib_only(self):
        code = ("import sys; sys.path.insert(0, %r);"
                "from boldt_embed import teacher_calibration;"
                "assert 'torch' not in sys.modules; print('clean')") % str(ROOT / "src")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


if __name__ == "__main__":
    unittest.main()
