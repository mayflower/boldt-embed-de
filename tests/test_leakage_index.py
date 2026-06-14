"""Tests for the scalable leakage scanner (src/boldt_embed/leakage_index.py). Pure stdlib.

Covers: exact / normalized / near-duplicate detection, unrelated-not-flagged, and that the
two-stage blocking keeps the scan SUBQUADRATIC (verify-stage comparisons << n_train*n_eval).
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

from boldt_embed import leakage_index as li  # noqa: E402

FIX = ROOT / "tests" / "fixtures"


def _index(eval_texts):
    units = [(f"e{i}", "evalset", "text", t) for i, t in enumerate(eval_texts)]
    return li.build_eval_leakage_index(units)


def _scan(cands, eval_texts, **kw):
    return li.find_candidate_leakage(cands, _index(eval_texts), **kw)


class TestDetection(unittest.TestCase):
    def test_exact_duplicate_detected(self):
        t = "Die Mietkaution darf hoechstens drei Nettokaltmieten betragen und ist zu verzinsen."
        res = _scan([{"id": "c", "document": t}], [t])
        self.assertEqual(len(res["hits"]), 1)
        self.assertEqual(res["hits"][0]["kind"], "exact")

    def test_normalized_duplicate_detected(self):
        t = "Die Kuendigungsfrist betraegt drei Monate und muss schriftlich erfolgen."
        res = _scan([{"id": "c", "document": t.upper() + "!!!"}], [t])
        self.assertEqual(len(res["hits"]), 1)
        self.assertEqual(res["hits"][0]["kind"], "exact_normalized")

    def test_near_duplicate_detected(self):
        base = " ".join(f"wort{j}" for j in range(40))
        near = base + " zusatz ende"          # +2 tokens -> Jaccard ~0.95
        res = _scan([{"id": "c", "document": near}], [base])
        self.assertEqual(len(res["hits"]), 1)
        self.assertEqual(res["hits"][0]["kind"], "near_duplicate")
        self.assertGreaterEqual(res["hits"][0]["score"], 0.9)

    def test_unrelated_not_flagged(self):
        eval_t = "Die Stadt Regensburg liegt an der Donau in Bayern und ist sehr alt."
        cand = "Kartoffeln werden in Salzwasser etwa zwanzig Minuten lang gekocht und abgegossen."
        res = _scan([{"id": "c", "document": cand}], [eval_t])
        self.assertEqual(res["hits"], [])

    def test_query_field_also_checked(self):
        t = "Wer regierte das Heilige Roemische Reich im Jahr 1500 nach Christus genau?"
        res = _scan([{"id": "c", "query": t, "document": "voellig anderer text hier"}], [t])
        self.assertEqual(len(res["hits"]), 1)
        self.assertEqual(res["hits"][0]["candidate_field"], "query")


class TestSubquadratic(unittest.TestCase):
    def test_blocking_keeps_comparisons_subquadratic(self):
        # N distinct eval texts + N distinct candidates (all unrelated) + 1 planted exact leak.
        N = 300
        evals = [" ".join(f"e{i}_{j}" for j in range(14)) for i in range(N)]
        cands = [{"id": f"c{i}", "document": " ".join(f"c{i}_{j}" for j in range(14))}
                 for i in range(N)]
        cands[7]["document"] = evals[42]          # one real leak
        res = li.find_candidate_leakage(cands, _index(evals))
        naive = N * N
        comps = res["stats"]["jaccard_comparisons"]
        # the planted leak is found ...
        self.assertTrue(any(h["candidate_id"] == "c7" for h in res["hits"]))
        # ... and we did vastly fewer than the naive n*m exact comparisons.
        self.assertLess(comps, N, f"expected << {naive}, did {comps}")


class TestFixturesAndCli(unittest.TestCase):
    def test_fixture_scan_classes(self):
        cands = [json.loads(l) for l in (FIX / "candidates_leakage.jsonl").read_text("utf-8").splitlines()]
        evals = [json.loads(l)["text"] for l in (FIX / "eval_leakage.jsonl").read_text("utf-8").splitlines()]
        res = _scan(cands, evals)
        kinds = sorted(h["kind"] for h in res["hits"])
        self.assertIn("exact", kinds)
        self.assertIn("exact_normalized", kinds)
        self.assertIn("near_duplicate", kinds)
        flagged = {h["candidate_id"] for h in res["hits"]}
        self.assertNotIn("c4_unrelated", flagged)

    def test_cli_runs_and_cleans(self):
        with tempfile.TemporaryDirectory() as d:
            rep, hits, clean = (pathlib.Path(d) / x for x in ("r.json", "h.jsonl", "clean.jsonl"))
            out = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "run_full_leakage_scan.py"),
                 "--candidates", str(FIX / "candidates_leakage.jsonl"),
                 "--eval-corpus", str(FIX / "eval_leakage.jsonl"),
                 "--output", str(rep), "--hits-output", str(hits), "--drop-hits", str(clean)],
                capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            report = json.loads(rep.read_text("utf-8"))
            self.assertEqual(report["n_eval_texts"], 3)
            self.assertEqual(report["total_flagged_candidates"], 4)
            self.assertLess(report["jaccard_comparisons"], report["naive_comparisons"])
            # cleaned file holds only the unrelated candidate
            kept = [json.loads(l) for l in clean.read_text("utf-8").splitlines()]
            self.assertEqual([c["id"] for c in kept], ["c4_unrelated"])

    def test_module_is_stdlib_only(self):
        code = ("import sys; sys.path.insert(0, %r);"
                "from boldt_embed import leakage_index;"
                "assert 'torch' not in sys.modules; print('clean')") % str(ROOT / "src")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


class TestTrainingGate(unittest.TestCase):
    def test_clean_report_is_acceptable(self):
        self.assertTrue(li.leakage_report_is_clean(
            {"exact_hits": 0, "exact_normalized_hits": 0, "near_duplicate_hits": 0}))

    def test_dirty_report_not_acceptable(self):
        self.assertFalse(li.leakage_report_is_clean(
            {"exact_hits": 3, "near_duplicate_hits": 1}))

    def test_cleaned_report_is_acceptable(self):
        self.assertTrue(li.leakage_report_is_clean(
            {"exact_hits": 3, "cleaned_candidates_path": "/tmp/clean.jsonl"}))

    def test_require_raises_on_missing_and_dirty(self):
        with self.assertRaises(ValueError):
            li.require_clean_leakage_report("/nonexistent/leakage_report.json")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"exact_hits": 2, "near_duplicate_hits": 0}, f)
            dirty = f.name
        with self.assertRaises(ValueError):
            li.require_clean_leakage_report(dirty)

    def test_require_passes_on_clean(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"exact_hits": 0, "near_duplicate_hits": 0}, f)
            clean = f.name
        li.require_clean_leakage_report(clean)  # must not raise


if __name__ == "__main__":
    unittest.main()
