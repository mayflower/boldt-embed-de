"""Tests for the v3 real-domain acquisition framework (fail-closed). Pure stdlib, no network."""
import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import domain_source_acquisition as dsa  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "data_sources_v3.json"
SCRIPT = ROOT / "scripts" / "acquire_v3_sources.py"
SHIPPED = ROOT / "configs" / "data_sources_v3.json"


def _base():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class TestManifestValidation(unittest.TestCase):
    def test_fixture_and_shipped_manifests_valid(self):
        self.assertEqual(dsa.validate_v3_manifest(_base()), [])
        self.assertEqual(dsa.validate_v3_manifest(json.loads(SHIPPED.read_text("utf-8"))), [])

    def test_unverified_license_cannot_train(self):
        d = _base()
        d["sources"][1]["allowed_for_training"] = True   # admin placeholder, license_verified=false
        errs = dsa.validate_v3_manifest(d)
        self.assertTrue(any("license_verified=true" in e for e in errs), errs)

    def test_public_benchmark_cannot_be_training_source(self):
        d = _base()
        d["sources"][2]["allowed_for_training"] = True   # germanquad, public_benchmark=true
        d["sources"][2]["public_benchmark"] = True
        d["sources"][2]["eval_only"] = False
        d["sources"][2]["contains_eval_overlap_risk"] = False
        errs = dsa.validate_v3_manifest(d)
        self.assertTrue(any("public_benchmark source cannot be a training source" in e for e in errs), errs)

    def test_overlap_risk_blocks_training(self):
        d = _base()
        s = d["sources"][0]                              # verified faq source
        s["contains_eval_overlap_risk"] = True
        errs = dsa.validate_v3_manifest(d)
        self.assertTrue(any("contains_eval_overlap_risk" in e for e in errs), errs)

    def test_uncertain_license_with_verified_flag_rejected(self):
        d = _base()
        d["sources"][0]["license"] = "unknown"
        errs = dsa.validate_v3_manifest(d)
        self.assertTrue(any("uncertain" in e for e in errs), errs)

    def test_bad_source_type_and_domain(self):
        d = _base()
        d["sources"][0]["source_type"] = "scrape"
        d["sources"][0]["domain"] = "made_up"
        errs = dsa.validate_v3_manifest(d)
        self.assertTrue(any("source_type" in e for e in errs))
        self.assertTrue(any("domain" in e for e in errs))


class TestRowValidation(unittest.TestCase):
    def test_valid_rows(self):
        self.assertEqual(dsa.validate_local_jsonl_row(
            {"id": "1", "query": "frage", "document": "doc", "source_id": "s", "license": "CC-BY-4.0"}), [])
        self.assertEqual(dsa.validate_local_jsonl_row(
            {"doc_id": "d1", "text": "ein dokument", "source_id": "s", "license": "CC0-1.0",
             "url": "http://x", "title": "t"}), [])

    def test_invalid_rows(self):
        self.assertTrue(dsa.validate_local_jsonl_row({"query": "q", "document": "d"}))   # no id/source/license
        self.assertTrue(dsa.validate_local_jsonl_row({"id": "1", "source_id": "s", "license": "x"}))  # no text/pair


class TestAcquire(unittest.TestCase):
    def test_dry_run_no_materialization(self):
        entries = dsa.load_v3_manifest(FIXTURE)
        with tempfile.TemporaryDirectory() as d:
            summary = dsa.acquire(entries, d, "dry-run")
            self.assertEqual(summary["status"], "ok")
            # nothing written in dry-run
            self.assertEqual(list(pathlib.Path(d).glob("*.jsonl")), [])
            self.assertIn("admin_real", summary["real_domains_missing"])

    def test_blocked_source_not_materialized(self):
        entries = dsa.load_v3_manifest(FIXTURE)
        with tempfile.TemporaryDirectory() as d:
            # give the BLOCKED admin placeholder a real on-disk file ...
            adminp = pathlib.Path(d) / "admin.jsonl"
            adminp.write_text(json.dumps({"id": "a1", "query": "q", "document": "doc",
                                          "source_id": "admin_real_placeholder",
                                          "license": "x"}) + "\n", encoding="utf-8")
            for e in entries:
                if e.source_id == "admin_real_placeholder":
                    e.loader["path"] = str(adminp)
            summary = dsa.acquire(entries, d, "materialize-local")
            blocked_ids = {b["source_id"] for b in summary["blocked"]}
            self.assertIn("admin_real_placeholder", blocked_ids)   # ... still NOT materialized
            self.assertNotIn("admin_real_placeholder", summary["rows_by_source"])
            self.assertFalse((pathlib.Path(d) / "admin_real_placeholder.jsonl").exists())

    def test_local_jsonl_source_materializes(self):
        entries = dsa.load_v3_manifest(FIXTURE)
        with tempfile.TemporaryDirectory() as d:
            faqp = pathlib.Path(d) / "faq.jsonl"
            with faqp.open("w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "f1", "query": "Wie beantrage ich Wohngeld?",
                                    "document": "Wohngeld beantragt man beim Amt.",
                                    "source_id": "faq_real_local", "license": "CC-BY-4.0"}) + "\n")
                f.write(json.dumps({"id": "bad"}) + "\n")    # invalid -> dropped
            for e in entries:
                if e.source_id == "faq_real_local":
                    e.loader["path"] = str(faqp)
            summary = dsa.acquire(entries, d, "materialize-local")
            self.assertEqual(summary["rows_by_source"].get("faq_real_local"), 1)   # 1 valid kept
            self.assertEqual(summary["real_domain_coverage"]["faq_real"], 1)
            self.assertTrue((pathlib.Path(d) / "faq_real_local.jsonl").exists())

    def test_supplemental_not_counted_as_real(self):
        entries = dsa.load_v3_manifest(FIXTURE)
        with tempfile.TemporaryDirectory() as d:
            sp = pathlib.Path(d) / "syn.jsonl"
            sp.write_text(json.dumps({"id": "s1", "query": "q", "document": "doc",
                                      "source_id": "synthetic_stress", "license": "x"}) + "\n",
                          encoding="utf-8")
            for e in entries:
                if e.source_id == "synthetic_stress":
                    e.loader["path"] = str(sp)
            summary = dsa.acquire(entries, d, "materialize-local")
            self.assertIn("synthetic_stress", summary["supplemental_sources"])
            # german_stress is not a *_real domain, and the source is supplemental either way
            self.assertEqual(summary["real_domain_coverage"],
                             {"faq_real": 0, "admin_real": 0,
                              "legal_adjacency_real_no_eval_overlap": 0})


class TestCli(unittest.TestCase):
    def _run(self, *extra):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--manifest", str(FIXTURE),
             "--output-dir", "/tmp/_v3_acq", "--mode", "dry-run", *extra],
            capture_output=True, text=True)

    def test_dry_run_exits_zero_and_no_network(self):
        out = self._run()
        self.assertEqual(out.returncode, 0, out.stderr)
        summary = json.loads(out.stdout)
        self.assertEqual(summary["status"], "ok")

    def test_fail_on_unverified_license_flag(self):
        # fixture has an unverified placeholder -> strict gate must fail
        out = self._run("--fail-on-unverified-license")
        self.assertEqual(out.returncode, 1)
        self.assertIn("unverified", (out.stdout + out.stderr).lower())

    def test_module_is_stdlib_only(self):
        code = ("import sys; sys.path.insert(0, %r);"
                "from boldt_embed import domain_source_acquisition;"
                "assert 'torch' not in sys.modules and 'datasets' not in sys.modules;"
                "print('clean')") % str(ROOT / "src")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


if __name__ == "__main__":
    unittest.main()
