"""Tests for the on-policy hard-negative refresh (stdlib only — no torch/transformers/GPU).

Covers: false-negative filter behaviour (a candidate within margin/ratio of the positive is
dropped); the real BM25-only path produces BOTH output files plus a manifest with kept/dropped
counts; a dense_teacher pool without artifacts fails closed with a clear message; and a provided
teacher-scores file is loaded and used.
"""
import argparse
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


R = _load("ar_refresh_hardnegatives")


def _args(config, out, **kw):
    ns = argparse.Namespace(config=str(config), out=str(out), teacher_scores=None,
                            max_records=None, dry_run=False, format="json", timestamp="T0")
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# A small fixture: distinct German query/positive pairs whose positives share enough vocabulary
# that BM25 surfaces the *other* positives as candidate (hard) negatives.
FIXTURE = [
    {"query_id": "q1", "query": "Wie hoch ist die Mehrwertsteuer in Deutschland",
     "document": "Die Mehrwertsteuer in Deutschland betraegt neunzehn Prozent.",
     "domain": "legal"},
    {"query_id": "q2", "query": "Welche Mehrwertsteuer gilt fuer Lebensmittel",
     "document": "Fuer Lebensmittel gilt in Deutschland eine ermaessigte Mehrwertsteuer.",
     "domain": "legal"},
    {"query_id": "q3", "query": "Hauptstadt von Frankreich",
     "document": "Paris ist die Hauptstadt von Frankreich.",
     "domain": "geo"},
    {"query_id": "q4", "query": "Hauptstadt von Italien",
     "document": "Rom ist die Hauptstadt von Italien.",
     "domain": "geo"},
]


def _base_config(tmp, **overrides):
    qpath = tmp / "train.jsonl"
    _write_jsonl(qpath, FIXTURE)
    cfg = {
        "name": "test_refresh",
        "queries": str(qpath),
        "corpus": str(qpath),
        "candidate_pools": ["bm25"],
        "teacher_scores": None,
        "negatives_per_query": 4,
        "candidate_list_k": 16,
        "false_negative_filter": {"method": "margin_or_ratio", "margin": 0.1, "ratio": 0.95},
        "domain_balance": {"max_per_domain": 4},
    }
    cfg.update(overrides)
    cpath = tmp / "config.json"
    cpath.write_text(json.dumps(cfg), encoding="utf-8")
    return cpath, qpath


class FalseNegativeFilterTests(unittest.TestCase):
    def test_effective_margin_folds_ratio(self):
        # margin_or_ratio: effective margin = max(margin, pos*(1-ratio)).
        eff = R.effective_margin(0.9, "margin_or_ratio", 0.05, 0.95)
        self.assertAlmostEqual(eff, 0.05)  # max(0.05, 0.9*0.05=0.045) -> margin dominates
        eff2 = R.effective_margin(0.9, "margin_or_ratio", 0.01, 0.95)
        self.assertAlmostEqual(eff2, 0.9 * 0.05)  # 0.045 > 0.01 -> ratio dominates
        # plain margin method ignores ratio.
        self.assertAlmostEqual(R.effective_margin(0.9, "margin", 0.2, 0.95), 0.2)
        # no pos score -> fall back to plain margin.
        self.assertAlmostEqual(R.effective_margin(None, "margin_or_ratio", 0.2, 0.95), 0.2)

    def test_candidate_within_margin_is_dropped(self):
        # Build a fixture where q1's BM25 candidate (q2) scores within margin of the positive,
        # so the teacher-gated filter drops it. We supply teacher scores making it a clear
        # false negative for q1 and a clear true negative for q3.
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            cfg_path, qpath = _base_config(tmp)
            teacher = tmp / "teacher.jsonl"
            _write_jsonl(teacher, [
                # q1 positive is its own doc; its candidate negative (q2 doc) scores ~equal ->
                # false negative -> dropped.
                {"query_id": "q1", "doc_id": "q1", "reranker_score": 0.90},
                {"query_id": "q1", "doc_id": "q2", "reranker_score": 0.88},
                # q3 positive vs candidate q4 doc scores far below -> kept (true negative).
                {"query_id": "q3", "doc_id": "q3", "reranker_score": 0.95},
                {"query_id": "q3", "doc_id": "q4", "reranker_score": 0.20},
            ])
            out = tmp / "out"
            manifest, rc = R.run(_args(cfg_path, out, teacher_scores=str(teacher)), now="T0")
            self.assertEqual(rc, 0)
            fs = manifest["filter_statistics"]
            # at least one candidate dropped as a likely false negative.
            self.assertGreaterEqual(fs["dropped"], 1)
            self.assertIn("within_margin_of_positive", fs["dropped_by_reason"])
            # the q1->q2 negative must NOT appear in q1's kept hard negatives.
            hn = [json.loads(l) for l in
                  (out / "hardnegatives.jsonl").read_text(encoding="utf-8").splitlines() if l]
            q1row = next(r for r in hn if r["query_id"] == "q1")
            self.assertNotIn("q2", [n["doc_id"] for n in q1row["negatives"]])


class BM25RealPathTests(unittest.TestCase):
    def test_bm25_only_produces_both_outputs_and_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            cfg_path, _ = _base_config(tmp)
            out = tmp / "out"
            manifest, rc = R.run(_args(cfg_path, out), now="2026-06-25T00:00:00+00:00")
            self.assertEqual(rc, 0)
            # both JSONL outputs exist and have one row per query.
            hn_path = out / "hardnegatives.jsonl"
            lw_path = out / "listwise_candidates.jsonl"
            self.assertTrue(hn_path.exists() and lw_path.exists())
            hn = [json.loads(l) for l in hn_path.read_text(encoding="utf-8").splitlines() if l]
            lw = [json.loads(l) for l in lw_path.read_text(encoding="utf-8").splitlines() if l]
            self.assertEqual(len(hn), len(FIXTURE))
            self.assertEqual(len(lw), len(FIXTURE))
            # listwise rows carry the positive as label-1 candidate.
            self.assertTrue(any(c["label"] == 1 for c in lw[0]["candidates"]))
            # manifest carries kept/dropped + per-domain balance + input hashes + filter params.
            self.assertIn("filter_statistics", manifest)
            self.assertIn("kept", manifest["filter_statistics"])
            self.assertIn("dropped", manifest["filter_statistics"])
            self.assertIn("per_domain_balance", manifest)
            self.assertIn("sha256", manifest["inputs"]["queries"])
            self.assertEqual(manifest["false_negative_filter"]["method"], "margin_or_ratio")
            self.assertEqual(manifest["timestamp_utc"], "2026-06-25T00:00:00+00:00")
            # report.md mentions filter statistics.
            report = (out / "report.md").read_text(encoding="utf-8")
            self.assertIn("Filter statistics", report)
            # with no teacher scores, nothing is dropped as a false negative.
            self.assertEqual(manifest["filter_statistics"]["dropped"], 0)

    def test_dry_run_writes_plan_no_mining(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            cfg_path, _ = _base_config(tmp)
            out = tmp / "out"
            manifest, rc = R.run(_args(cfg_path, out, dry_run=True), now="T0")
            self.assertEqual(rc, 0)
            self.assertTrue(manifest["dry_run"])
            self.assertIn("plan", manifest)
            self.assertNotIn("filter_statistics", manifest)
            # JSONL outputs exist but are empty on a dry run.
            self.assertEqual((out / "hardnegatives.jsonl").read_text(encoding="utf-8").strip(), "")


class FailClosedTests(unittest.TestCase):
    def test_dense_teacher_without_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            cfg_path, _ = _base_config(tmp, candidate_pools=["bm25", "dense_teacher"])
            out = tmp / "out"
            with self.assertRaises(R.FailClosed) as ctx:
                R.run(_args(cfg_path, out), now="T0")
            msg = str(ctx.exception)
            self.assertIn("dense_teacher", msg)
            self.assertIn("dense_teacher_embeddings", msg)

    def test_dense_teacher_cli_exit_code_2(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            cfg_path, _ = _base_config(tmp, candidate_pools=["dense_current"])
            out = tmp / "out"
            rc = R.main(["--config", str(cfg_path), "--out", str(out)])
            self.assertEqual(rc, 2)

    def test_missing_queries_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            cfg_path, _ = _base_config(tmp, queries=str(tmp / "does_not_exist.jsonl"))
            out = tmp / "out"
            with self.assertRaises(R.FailClosed) as ctx:
                R.run(_args(cfg_path, out), now="T0")
            self.assertIn("queries file missing", str(ctx.exception))


class TeacherScoresTests(unittest.TestCase):
    def test_provided_teacher_scores_are_loaded(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            cfg_path, _ = _base_config(tmp)
            teacher = tmp / "teacher.jsonl"
            _write_jsonl(teacher, [
                {"query_id": "q1", "doc_id": "q1", "embedding_score": 0.8, "reranker_score": 0.9},
                {"query_id": "q3", "doc_id": "q3", "embedding_score": 0.7, "reranker_score": 0.85},
            ])
            out = tmp / "out"
            manifest, rc = R.run(_args(cfg_path, out, teacher_scores=str(teacher)), now="T0")
            self.assertEqual(rc, 0)
            # the manifest records that teacher scores were present and loaded from the file.
            self.assertTrue(manifest["false_negative_filter"]["teacher_scores_present"])
            self.assertTrue(manifest["inputs"]["teacher_scores"]["exists"])

    def test_missing_teacher_scores_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            cfg_path, _ = _base_config(tmp)
            out = tmp / "out"
            with self.assertRaises(R.FailClosed) as ctx:
                R.run(_args(cfg_path, out, teacher_scores=str(tmp / "nope.jsonl")), now="T0")
            self.assertIn("teacher-scores file missing", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
