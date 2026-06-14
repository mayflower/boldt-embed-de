"""Tests for the v3 candidate builder (pure stdlib). Real-domain coverage + provenance + safety."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from boldt_embed import domain_source_acquisition as dsa  # noqa: E402
from boldt_embed import leakage_index as li  # noqa: E402
import build_v3_candidates as B  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
SCRIPT = ROOT / "scripts" / "build_v3_candidates.py"

_REQUIRED_ROW_FIELDS = ("query_id", "doc_id", "query", "document", "source_id", "source",
                        "domain", "license", "license_origin", "allowed_for_training",
                        "synthetic", "text_hash", "pair_hash")


def _src(source_id, domain, stype="local_jsonl", **over):
    s = {"source_id": source_id, "display_name": source_id, "domain": domain,
         "source_type": stype, "license": "CC-BY-4.0", "license_url": "http://x",
         "license_verified": True, "allowed_for_training": True, "eval_only": False,
         "public_benchmark": False, "contains_eval_overlap_risk": False,
         "requires_attribution": False, "supplemental": False, "notes": "",
         "loader": {"kind": stype, "path": f"{source_id}.jsonl"}, "expected_fields": {}}
    s.update(over)
    return s


def _setup(tmp, sources, raw_files):
    """Write a manifest + raw files; return (entries, raw_dir)."""
    man = pathlib.Path(tmp) / "manifest.json"
    man.write_text(json.dumps({"sources": sources}), encoding="utf-8")
    raw = pathlib.Path(tmp) / "raw"
    raw.mkdir()
    for sid, rows in raw_files.items():
        (raw / f"{sid}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return dsa.load_v3_manifest(man), raw


class TestBuild(unittest.TestCase):
    def test_provenance_fields_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            entries, raw = _setup(d, [_src("faq_real_local", "faq_real")],
                                  {"faq_real_local": [{"id": "1", "query": "Frage", "document": "Antwort Text"}]})
            res = B.build(entries, raw, {"faq_real": 1})
            row = res["candidates"][0]
            for f in _REQUIRED_ROW_FIELDS:
                self.assertIn(f, row, f)
            self.assertEqual(row["source_id"], "faq_real_local")
            self.assertEqual(row["license"], "CC-BY-4.0")
            self.assertEqual(row["license_origin"], "manifest")
            self.assertFalse(row["synthetic"])

    def test_quota_enforcement(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [{"id": str(i), "query": f"Frage {i}", "document": f"Antwort {i}"} for i in range(6)]
            entries, raw = _setup(d, [_src("faq_real_local", "faq_real")], {"faq_real_local": rows})
            ok = B.build(entries, raw, {"faq_real": 3})
            self.assertTrue(ok["report"]["quota"]["by_domain"]["faq_real"]["achieved"])
            self.assertEqual(ok["report"]["quota"]["missed"], [])
            miss = B.build(entries, raw, {"faq_real": 50})
            self.assertFalse(miss["report"]["quota"]["by_domain"]["faq_real"]["achieved"])
            self.assertTrue(any(m["domain"] == "faq_real" for m in miss["report"]["quota"]["missed"]))

    def test_public_benchmark_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            gq = _src("germanquad_eval", "wiki_non_eval", stype="hf_dataset",
                      allowed_for_training=False, eval_only=True, public_benchmark=True,
                      contains_eval_overlap_risk=True)
            entries, raw = _setup(d, [_src("faq_real_local", "faq_real"), gq],
                                  {"faq_real_local": [{"id": "1", "query": "q", "document": "doc"}],
                                   "germanquad_eval": [{"id": "g", "query": "leak", "document": "benchmark doc"}]})
            res = B.build(entries, raw, {"faq_real": 1})
            blocked = {b["source_id"]: b["reason"] for b in res["report"]["blocked_sources"]}
            self.assertEqual(blocked.get("germanquad_eval"), "public_benchmark")
            self.assertNotIn("germanquad_eval", {c["source_id"] for c in res["candidates"]})

    def test_leakage_hit_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            leak_doc = "Dieses Dokument ist exakt eine Evaluationspassage und darf nicht trainiert werden."
            rows = [{"id": "1", "query": "frage eins", "document": leak_doc},
                    {"id": "2", "query": "frage zwei", "document": "ein voellig anderes harmloses dokument hier"}]
            entries, raw = _setup(d, [_src("faq_real_local", "faq_real")], {"faq_real_local": rows})
            idx = li.build_eval_leakage_index([("e1", "eval", "text", leak_doc)])
            res = B.build(entries, raw, {"faq_real": 10}, leakage_index=idx)
            self.assertGreaterEqual(res["report"]["dropped_by_reason"]["leakage"], 1)
            self.assertTrue(all(leak_doc not in c["document"] for c in res["candidates"]))

    def test_pii_hit_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [{"id": "1", "query": "kontakt", "document": "Schreiben Sie an max@example.de bitte."},
                    {"id": "2", "query": "info", "document": "Ein harmloses Dokument ohne PII."}]
            entries, raw = _setup(d, [_src("faq_real_local", "faq_real")], {"faq_real_local": rows})
            res = B.build(entries, raw, {"faq_real": 10}, pii_scan=True)
            self.assertGreaterEqual(res["report"]["dropped_by_reason"]["pii"], 1)

    def test_document_only_no_fake_query(self):
        with tempfile.TemporaryDirectory() as d:
            legal = _src("legal_corpus", "legal_adjacency_real_no_eval_overlap",
                         stype="local_corpus_jsonl")
            entries, raw = _setup(d, [legal],
                                  {"legal_corpus": [{"id": "l1", "text": "Nach Paragraph 573 BGB gilt Folgendes."},
                                                    {"id": "l2", "text": "Eine weitere rechtliche Passage."}]})
            res = B.build(entries, raw, {"legal_adjacency_real_no_eval_overlap": 5})
            self.assertEqual(len(res["candidates"]), 0)          # NO fabricated pairs
            self.assertEqual(res["report"]["totals"]["passages"], 2)
            for p in res["passages"]:
                self.assertEqual(p["record_type"], "passage")
                self.assertNotIn("query", p)
            # the real-domain pair quota is correctly MISSED (document-only != pairs)
            self.assertTrue(any(m["domain"] == "legal_adjacency_real_no_eval_overlap"
                                for m in res["report"]["quota"]["missed"]))

    def test_synthetic_not_counted_as_real(self):
        with tempfile.TemporaryDirectory() as d:
            syn = _src("syn_faq", "faq_real", license="synthetic-inherits-source", supplemental=True)
            entries, raw = _setup(d, [syn],
                                  {"syn_faq": [{"id": "1", "query": "frage", "document": "doc"}]})
            res = B.build(entries, raw, {"faq_real": 1})
            self.assertTrue(res["candidates"][0]["synthetic"])
            q = res["report"]["quota"]["by_domain"]["faq_real"]
            self.assertEqual(q["real"], 0)           # synthetic does not count toward the real target
            self.assertFalse(q["achieved"])


class TestCli(unittest.TestCase):
    def test_listed_dry_run_passes(self):
        out = subprocess.run(
            [sys.executable, str(SCRIPT), "--manifest", str(FIX / "data_sources_v3.json"),
             "--config", str(FIX / "v3_real_domain_generalization.json"),
             "--raw-dir", str(FIX / "raw_v3"), "--output", "/tmp/_v3c.jsonl",
             "--target-count", "100", "--dry-run"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(json.loads(out.stdout)["status"], "ok")

    def test_real_build_requires_leakage_index(self):
        # NOT dry-run + no --leakage-index -> hard error (exit 2)
        out = subprocess.run(
            [sys.executable, str(SCRIPT), "--manifest", str(FIX / "data_sources_v3.json"),
             "--config", str(FIX / "v3_real_domain_generalization.json"),
             "--raw-dir", str(FIX / "raw_v3"), "--output", "/tmp/_v3c.jsonl",
             "--target-count", "100"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 2)
        self.assertIn("leakage-index required", out.stderr)

    def test_license_failure_invalid_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            bad = pathlib.Path(d) / "m.json"
            bad.write_text(json.dumps({"sources": [
                _src("x", "faq_real", license_verified=False)]}), encoding="utf-8")  # allowed+unverified
            out = subprocess.run(
                [sys.executable, str(SCRIPT), "--manifest", str(bad),
                 "--config", str(FIX / "v3_real_domain_generalization.json"),
                 "--raw-dir", str(FIX / "raw_v3"), "--output", "/tmp/_v3c.jsonl", "--dry-run"],
                capture_output=True, text=True)
            self.assertEqual(out.returncode, 2)
            self.assertIn("license_verified", out.stderr)

    def test_module_is_stdlib_only(self):
        code = ("import sys; sys.path.insert(0, %r); sys.path.insert(0, %r);"
                "import build_v3_candidates;"
                "assert 'torch' not in sys.modules; print('clean')") % (str(ROOT / "src"), str(ROOT / "scripts"))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


if __name__ == "__main__":
    unittest.main()
