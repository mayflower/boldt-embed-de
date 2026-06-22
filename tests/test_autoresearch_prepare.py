"""Tests for the AutoResearch preparation manifest builder (stdlib, no ML, no network)."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("ar_prepare")


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")


class PrepareUnitTests(unittest.TestCase):
    def test_jsonl_count_and_cap(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "t.jsonl"
            _write_jsonl(p, [{"a": i} for i in range(5)])
            self.assertEqual(len(P.read_jsonl(p)), 5)
            self.assertEqual(len(P.read_jsonl(p, max_records=2)), 2)

    def test_sha256(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "f.bin"
            p.write_bytes(b"hello-boldt")
            self.assertEqual(P.sha256_file(p), hashlib.sha256(b"hello-boldt").hexdigest())

    def test_required_field_and_breakdown_counts(self):
        records = [
            {"query_id": "q1", "query": "a", "document": "d", "source": "webfaq_train",
             "domain": "faq", "license": "cc"},
            {"query_id": "q2", "query": "", "document": "d", "source": "mmarco_de",
             "domain": "web", "license": "apache"},   # missing query (empty)
            {"query_id": "q3", "document": "d", "source": "webfaq_train"},  # missing query
        ]
        s = P.summarize_train(records)
        self.assertEqual(s["count"], 3)
        self.assertEqual(s["missing_required"]["query"], 2)
        self.assertEqual(s["records_missing_any_required"], 2)
        self.assertEqual(s["source_counts"]["webfaq_train"], 2)
        self.assertEqual(s["domain_counts"]["faq"], 1)
        self.assertEqual(s["license_counts"]["apache"], 1)

    def test_leakage_hit_extraction(self):
        self.assertEqual(P.extract_leakage_hits({"hits": 0}), 0)
        self.assertEqual(P.extract_leakage_hits({"num_hits": 3}), 3)
        self.assertEqual(P.extract_leakage_hits({"leakage_hits": 5}), 5)
        self.assertEqual(P.extract_leakage_hits({"summary": {"hits": 2}}), 2)
        self.assertIsNone(P.extract_leakage_hits({"unrelated": 1}))

    def test_leakage_index_report_shape(self):
        # the repo's real scan report (leakage_index) uses exact_*/near_duplicate fields
        report = {"exact_hits": 1, "exact_normalized_hits": 0, "near_duplicate_hits": 2}
        self.assertEqual(P.extract_leakage_hits(report), 3)
        self.assertTrue(P.looks_like_leakage_report(report))
        self.assertFalse(P.looks_like_leakage_report({"unrelated": 1}))

    def test_optional_eval_set_missing_ok_required_missing_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            present = pathlib.Path(d) / "webfaq.jsonl"
            _write_jsonl(present, [{"query_id": "q1"}])
            manifest = {"sets": [
                {"name": "webfaq", "role": "primary", "path": str(present)},
                {"name": "local_rag", "role": "primary_optional",
                 "path": str(pathlib.Path(d) / "missing_opt.jsonl"), "optional": True},
            ]}
            res = P.evaluate_eval_manifest(manifest)
            self.assertEqual(res["missing_required"], [])
            self.assertEqual(res["missing_optional"], ["local_rag"])

            manifest_bad = {"sets": [
                {"name": "webfaq", "role": "primary",
                 "path": str(pathlib.Path(d) / "nope.jsonl")},
            ]}
            self.assertEqual(P.evaluate_eval_manifest(manifest_bad)["missing_required"], ["webfaq"])


class PrepareMainTests(unittest.TestCase):
    def _args(self, **kw):
        defaults = dict(train=None, eval_manifest=None, baseline_model="ref", out=None,
                        max_records=None, seed=1337, require_leakage_report=None, format="json")
        defaults.update(kw)
        return argparse.Namespace(**defaults)

    def test_clean_leakage_promotable(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            train = d / "train.jsonl"
            _write_jsonl(train, [{"query_id": "q1", "query": "a", "document": "x"}])
            eval_present = d / "webfaq.jsonl"
            _write_jsonl(eval_present, [{"query_id": "q1"}])
            manifest = d / "eval_manifest.json"
            manifest.write_text(json.dumps({"sets": [
                {"name": "webfaq", "role": "primary", "path": str(eval_present)}]}),
                encoding="utf-8")
            leak = d / "leak.json"
            leak.write_text(json.dumps({"hits": 0}), encoding="utf-8")
            man, _, _, code = P.build_manifest(self._args(
                train=str(train), eval_manifest=str(manifest), out=str(d / "out"),
                require_leakage_report=str(leak)))
            self.assertEqual(code, 0)
            self.assertTrue(man["promotable"])
            self.assertEqual(man["leakage"]["status"], "clean")

    def test_leakage_hits_make_it_fail_and_not_promotable(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            train = d / "train.jsonl"
            _write_jsonl(train, [{"query_id": "q1", "query": "a", "document": "x"}])
            eval_present = d / "webfaq.jsonl"
            _write_jsonl(eval_present, [{"query_id": "q1"}])
            manifest = d / "eval_manifest.json"
            manifest.write_text(json.dumps({"sets": [
                {"name": "webfaq", "role": "primary", "path": str(eval_present)}]}),
                encoding="utf-8")
            leak = d / "leak.json"
            leak.write_text(json.dumps({"hits": 4}), encoding="utf-8")
            man, _, _, code = P.build_manifest(self._args(
                train=str(train), eval_manifest=str(manifest), out=str(d / "out"),
                require_leakage_report=str(leak)))
            self.assertEqual(code, 1)
            self.assertFalse(man["promotable"])
            self.assertEqual(man["leakage"]["status"], "leak_detected")

    def test_no_leakage_report_not_promotable(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            train = d / "train.jsonl"
            _write_jsonl(train, [{"query_id": "q1", "query": "a", "document": "x"}])
            eval_present = d / "webfaq.jsonl"
            _write_jsonl(eval_present, [{"query_id": "q1"}])
            manifest = d / "eval_manifest.json"
            manifest.write_text(json.dumps({"sets": [
                {"name": "webfaq", "role": "primary", "path": str(eval_present)}]}),
                encoding="utf-8")
            man, _, _, code = P.build_manifest(self._args(
                train=str(train), eval_manifest=str(manifest), out=str(d / "out")))
            self.assertEqual(code, 0)             # not fatal, just not promotable
            self.assertFalse(man["promotable"])
            self.assertEqual(man["leakage"]["status"], "not_checked")


if __name__ == "__main__":
    unittest.main()
