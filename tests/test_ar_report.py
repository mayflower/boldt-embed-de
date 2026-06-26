"""Tests for scripts/ar_report.py — builds a tiny fixture outputs/ tree in a tempdir and
points the report at it via the injectable --root / build_report(root=...). Stdlib only.

Asserts: a missing MTEB summary is marked 'missing' (never 0); best-by-task is correct;
promotable detection vs the same-size-peer aggregate; Pareto frontier membership;
events.jsonl absence handled; and both json + markdown render.
"""
import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("ar_report", "scripts/ar_report.py")


def _write_summary(mteb_dir, label, scores):
    d = mteb_dir / label
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({
        "meta": {"label": label, "tasks": list(scores)},
        "scores": scores,
    }), encoding="utf-8")


def _make_fixture(tmp):
    """A minimal but representative outputs/ tree under tmp."""
    root = pathlib.Path(tmp)
    out = root / "outputs"
    ar = out / "autoresearch"
    mteb = out / "mteb"
    ar.mkdir(parents=True, exist_ok=True)
    mteb.mkdir(parents=True, exist_ok=True)

    # --- same-size peers (the bar to beat). peer frontier = max per task.
    _write_summary(mteb, "e5-base", {
        "GermanQuAD-Retrieval": 0.92, "GerDaLIRSmall": 0.15,
        "MIRACLRetrievalHardNegatives": 0.53, "MultiLongDocRetrieval": 0.26})
    _write_summary(mteb, "lfm2.5", {
        "GermanQuAD-Retrieval": 0.90, "GerDaLIRSmall": 0.20,
        "MIRACLRetrievalHardNegatives": 0.50, "MultiLongDocRetrieval": 0.30})
    # peer frontier = [0.92, 0.20, 0.53, 0.30] -> mean = 0.4875

    # --- a strong candidate that BEATS the peer aggregate.
    _write_summary(mteb, "cand-strong", {
        "GermanQuAD-Retrieval": 0.95, "GerDaLIRSmall": 0.25,
        "MIRACLRetrievalHardNegatives": 0.60, "MultiLongDocRetrieval": 0.40})
    # mean = 0.55 > 0.4875 -> promotable

    # --- a weak candidate that does NOT beat the peer aggregate.
    _write_summary(mteb, "cand-weak", {
        "GermanQuAD-Retrieval": 0.80, "GerDaLIRSmall": 0.10,
        "MIRACLRetrievalHardNegatives": 0.40, "MultiLongDocRetrieval": 0.20})
    # mean = 0.375 < 0.4875 -> not promotable

    # --- a stretch reference (must be excluded from candidates).
    _write_summary(mteb, "qwen3-0.6b", {
        "GermanQuAD-Retrieval": 0.97, "GerDaLIRSmall": 0.30,
        "MIRACLRetrievalHardNegatives": 0.65, "MultiLongDocRetrieval": 0.45})

    # --- a dense run that has NO MTEB summary (must be 'missing', not 0).
    # results.tsv with two runs.
    header = ("timestamp_utc\tcommit\trun_id\tmode\tstatus\tscore\twebfaq_recall100\t"
              "webfaq_ndcg10\twebfaq_mrr10\tlocal_rag_recall100\tgermanquad_ndcg10\t"
              "dt_test_ndcg10\tm256_retention\tleakage_hits\tleakage_status\tbudget_minutes\t"
              "elapsed_seconds\tinvalid_for_default_loop\tvram_gb\tthroughput_pairs_per_sec\t"
              "config_path\tnotes")
    rows = [
        # run-A: best webfaq recall, decent throughput, low vram
        ("2026-06-22T18:00:00+00:00\tabc\trun-A\treal\tkeep\t-0.03\t0.98\t0.68\t0.65\t\t"
         "0.88\t0.97\t0.99\t0\tclean\t20\t256\tFalse\t3.77\t2400\t/cfg/A.json\tloop"),
        # run-B: lower webfaq recall (regression flag), higher vram, lower throughput
        ("2026-06-22T18:10:00+00:00\tabc\trun-B\treal\tkeep\t-0.06\t0.90\t0.66\t0.63\t\t"
         "0.87\t0.96\t0.98\t0\tclean\t20\t256\tFalse\t9.10\t800\t/cfg/B.json\tloop"),
    ]
    (ar / "results.tsv").write_text("\n".join([header] + rows) + "\n", encoding="utf-8")

    # NOTE: deliberately NO events.jsonl, NO runs/, NO manifests -> all must be 'missing'.
    return root


class ReportFixtureTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = _make_fixture(self._tmp.name)
        self.report = R.build_report(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_peer_aggregate_is_frontier_mean(self):
        # max per task over e5-base/lfm2.5 = [0.92,0.20,0.53,0.30] -> 0.4875
        self.assertAlmostEqual(self.report["peer_aggregate"], 0.4875, places=4)

    def test_stretch_model_excluded_from_candidates(self):
        labels = {c["label"] for c in self.report["candidates"]}
        self.assertNotIn("qwen3-0.6b", labels)
        self.assertNotIn("e5-base", labels)
        self.assertNotIn("lfm2.5", labels)

    def test_dense_run_without_mteb_marked_missing_not_zero(self):
        recs = {c["label"]: c for c in self.report["candidates"]}
        self.assertIn("run-A", recs)
        a = recs["run-A"]
        self.assertFalse(a["has_mteb"])
        # MTEB task metrics must be None (missing), NOT 0.0
        self.assertIsNone(a["metrics"]["MTEB_GermanQuAD"])
        self.assertIsNone(a["mteb_aggregate"])
        self.assertIn("run-A", self.report["missing"]["mteb_summaries_missing"])

    def test_dense_metrics_parsed_as_numbers(self):
        recs = {c["label"]: c for c in self.report["candidates"]}
        self.assertAlmostEqual(recs["run-A"]["metrics"]["webfaq_recall@100"], 0.98, places=4)
        self.assertAlmostEqual(recs["run-A"]["metrics"]["vram_gb"], 3.77, places=4)

    def test_best_by_task_correct(self):
        bbt = self.report["best_by_task"]
        # best webfaq recall is run-A (0.98 vs 0.90)
        self.assertEqual(bbt["webfaq_recall@100"]["label"], "run-A")
        # best MTEB GermanQuAD among candidates (peers/stretch excluded) is cand-strong (0.95)
        self.assertEqual(bbt["MTEB_GermanQuAD"]["label"], "cand-strong")
        # lowest vram is best -> run-A (3.77 < 9.10)
        self.assertEqual(bbt["vram_gb"]["label"], "run-A")
        # highest throughput -> run-A (2400 > 800)
        self.assertEqual(bbt["throughput_pairs_per_sec"]["label"], "run-A")

    def test_promotable_only_strong_candidate(self):
        promo = {p["label"] for p in self.report["beats_peer_aggregate"]}
        self.assertIn("cand-strong", promo)
        self.assertNotIn("cand-weak", promo)
        # dense-only runs have no MTEB aggregate -> cannot be promotable
        self.assertNotIn("run-A", promo)

    def test_pareto_frontier_membership(self):
        front = set(self.report["pareto_frontier"])
        # cand-strong dominates cand-weak on all comparable MTEB axes -> weak off frontier
        self.assertIn("cand-strong", front)
        self.assertNotIn("cand-weak", front)
        # run-A vs run-B: run-A is better on every comparable dense metric -> B dominated
        self.assertIn("run-A", front)
        self.assertNotIn("run-B", front)

    def test_events_and_manifests_missing(self):
        miss = self.report["missing"]
        self.assertEqual(miss["events_jsonl"]["status"], "missing")
        # no manifests on disk -> all three kinds reported missing
        self.assertIn("merge", miss["manifests"])
        self.assertIn("distill", miss["manifests"])
        self.assertIn("specialist", miss["manifests"])

    def test_regression_flagged_for_run_b(self):
        reasons = {r["label"] for r in self.report["regressions"]}
        # run-B webfaq recall (0.90) is >0.02 below best real run (0.98)
        self.assertIn("run-B", reasons)

    def test_missing_results_tsv_marked_missing_not_zero(self):
        with tempfile.TemporaryDirectory() as t:
            root = pathlib.Path(t)
            (root / "outputs" / "mteb").mkdir(parents=True)
            rep = R.build_report(root)
            self.assertEqual(rep["inputs"]["results_tsv"]["status"], "missing")
            self.assertEqual(rep["n_candidates"], 0)


class RenderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = _make_fixture(self._tmp.name)
        self.report = R.build_report(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_markdown_renders(self):
        md = R.render_markdown(self.report)
        self.assertIn("# AutoResearch frontier report", md)
        self.assertIn("Leaderboard", md)
        self.assertIn("cand-strong", md)
        self.assertIn("missing", md)   # MTEB-missing dense runs rendered as 'missing'

    def test_json_serialisable(self):
        s = json.dumps(self.report)
        self.assertIn("peer_aggregate", s)

    def test_leaderboard_tsv_renders(self):
        tsv = R.render_leaderboard_tsv(self.report)
        head = tsv.splitlines()[0].split("\t")
        self.assertIn("label", head)
        self.assertIn("mteb_aggregate", head)
        self.assertIn("MTEB_GermanQuAD", head)
        # the MTEB-missing dense run shows 'missing' in its MTEB columns
        line = next(l for l in tsv.splitlines() if l.startswith("run-A\t"))
        self.assertIn("missing", line)

    def test_write_reports_creates_three_files(self):
        written = R.write_reports(self.report, self.root)
        for k in ("json", "md", "tsv"):
            self.assertTrue(pathlib.Path(written[k]).exists())
        # round-trip the json
        data = json.loads(pathlib.Path(written["json"]).read_text(encoding="utf-8"))
        self.assertEqual(data["n_candidates"], self.report["n_candidates"])

    def test_main_markdown_runs(self):
        rc = R.main(["--root", str(self.root), "--format", "markdown", "--no-write"])
        self.assertEqual(rc, 0)

    def test_main_candidate_view_json(self):
        rc = R.main(["--root", str(self.root), "--candidate", "cand-strong",
                     "--format", "json", "--no-write"])
        self.assertEqual(rc, 0)

    def test_main_candidate_missing(self):
        rc = R.main(["--root", str(self.root), "--candidate", "does-not-exist",
                     "--format", "json", "--no-write"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
