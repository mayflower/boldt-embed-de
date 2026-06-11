"""Stdlib tests for the v2 candidate builder + domain-target sampler. No network, no ML."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data_pipeline as dp  # noqa: E402

SCRIPT = ROOT / "scripts" / "build_v2_candidates.py"
MANIFEST = ROOT / "tests" / "fixtures" / "v2_sources_manifest.json"
SEED = ROOT / "tests" / "fixtures" / "v2_candidates_seed.jsonl"
V2CFG = ROOT / "configs" / "experiments" / "v2_generalization.json"


def _run(args):
    return subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(MANIFEST),
                           "--domain-config", str(V2CFG)] + args, capture_output=True, text=True)


def _tmp_jsonl(rows):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    f.close()
    return f.name


class TestSamplerUnit(unittest.TestCase):
    def test_deterministic_and_capped(self):
        rows = [{"domain": "web", "i": i} for i in range(10)] + [{"domain": "faq", "i": i} for i in range(3)]
        a = dp.sample_to_domain_targets(rows, {"web": 4}, seed=1)
        b = dp.sample_to_domain_targets(rows, {"web": 4}, seed=1)
        self.assertEqual(a, b)                                   # deterministic
        self.assertEqual(sum(1 for r in a if r["domain"] == "web"), 4)  # capped
        self.assertEqual(sum(1 for r in a if r["domain"] == "faq"), 3)  # uncapped


class TestBuilderHappyPath(unittest.TestCase):
    def test_admits_allowed_rows(self):
        out = _run(["--source-jsonl", str(SEED), "--target-count", "20", "--dedup",
                    "--pii-scan", "--dry-run"])
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn('"admitted": 3', out.stdout)
        self.assertIn("[domains]", out.stdout)


class TestBuilderGates(unittest.TestCase):
    def test_disallowed_and_unknown_sources_blocked(self):
        rows = [
            {"source": "tiny_local_web", "domain": "web", "query": "gute frage hier", "document": "ein passender deutscher absatz dazu"},
            {"source": "tiny_eval_quad", "domain": "qa_wiki", "query": "eval q", "document": "eval doc"},   # eval -> not allowed
            {"source": "ghost_source", "domain": "web", "query": "x", "document": "y"},                       # unknown
        ]
        out = _run(["--source-jsonl", _tmp_jsonl(rows), "--target-count", "20", "--dry-run"])
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn('"blocked_not_allowed_for_training": 1', out.stdout)
        self.assertIn('"blocked_unknown_source": 1', out.stdout)
        self.assertIn('"admitted": 1', out.stdout)

    def test_pii_row_fails_without_allow(self):
        rows = [{"source": "tiny_local_web", "domain": "web", "query": "kontakt frage",
                 "document": "Schreiben Sie an max.mustermann@example.com fuer mehr infos bitte"}]
        out = _run(["--source-jsonl", _tmp_jsonl(rows), "--target-count", "20", "--pii-scan", "--dry-run"])
        self.assertEqual(out.returncode, 3, out.stdout + out.stderr)
        self.assertIn("PII found", out.stderr)

    def test_pii_row_dropped_with_allow(self):
        rows = [
            {"source": "tiny_local_web", "domain": "web", "query": "saubere frage hier", "document": "ein sauberer deutscher absatz ohne pii"},
            {"source": "tiny_local_web", "domain": "web", "query": "kontakt frage", "document": "mail an max@example.com bitte"},
        ]
        out = _run(["--source-jsonl", _tmp_jsonl(rows), "--target-count", "20", "--pii-scan",
                    "--allow-pii", "--dry-run"])
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("dropped 1 rows", out.stdout)

    def test_leakage_row_dropped(self):
        leak_doc = "Die Berliner Mauer wurde im Jahr 1961 errichtet und fiel 1989"
        rows = [
            {"source": "tiny_local_web", "domain": "web", "query": "wann mauer", "document": leak_doc},
            {"source": "tiny_local_web", "domain": "web", "query": "andere frage", "document": "ein voellig anderer deutscher absatz hier"},
        ]
        evalc = _tmp_jsonl([{"text": leak_doc}])
        out = _run(["--source-jsonl", _tmp_jsonl(rows), "--target-count", "20",
                    "--leakage-corpus-jsonl", evalc, "--dry-run"])
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("leaked_rows=1", out.stdout)


if __name__ == "__main__":
    unittest.main()
