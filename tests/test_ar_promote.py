"""P10 — MTEB promotion: ar_promote runs the real frontier gate over fixtures (pass/fail) + ar_mteb_trial."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ar_promote = _load("ar_promote")
ar_mteb_trial = _load("ar_mteb_trial")

TASKS = ["GermanQuAD-Retrieval", "GerDaLIRSmall", "MIRACLRetrievalHardNegatives", "MultiLongDocRetrieval"]


def _summary(root, label, scores):
    d = Path(root) / label
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({"meta": {"label": label}, "scores": scores}),
                                    encoding="utf-8")


def _fixture_root(d, cand_scores):
    # peers low, baseline low -> a high candidate beats the frontier and never regresses
    _summary(d, "e5-base", dict.fromkeys(TASKS, 0.50))
    _summary(d, "lfm2.5", dict.fromkeys(TASKS, 0.48))
    _summary(d, "v6-1-baseline-512", dict.fromkeys(TASKS, 0.40))
    _summary(d, "cand", dict(zip(TASKS, cand_scores)))
    return d


class TestPromoteGate(unittest.TestCase):
    def test_no_torch(self):
        import pathlib as _p
        sys.path.insert(0, str(_p.Path(__file__).resolve().parent))
        from torch_free import script_is_torch_free
        self.assertTrue(script_is_torch_free("scripts/ar_promote.py"))

    def test_candidate_beating_peers_is_promotable(self):
        with tempfile.TemporaryDirectory() as d:
            _fixture_root(d, [0.90, 0.90, 0.90, 0.90])
            v = ar_promote.run_gate("cand", "e5-base,lfm2.5", "v6-1-baseline-512", 0.005, mteb_root=d)
            self.assertTrue(v["promotable"], v)
            self.assertEqual(v["failed_gates"], [])

    def test_candidate_below_peers_not_promotable(self):
        with tempfile.TemporaryDirectory() as d:
            _fixture_root(d, [0.45, 0.45, 0.45, 0.45])
            v = ar_promote.run_gate("cand", "e5-base,lfm2.5", "v6-1-baseline-512", 0.005, mteb_root=d)
            self.assertFalse(v["promotable"])
            self.assertIn("beats_peer_frontier_aggregate", v["failed_gates"])

    def test_missing_baseline_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            _summary(d, "e5-base", dict.fromkeys(TASKS, 0.50))
            _summary(d, "lfm2.5", dict.fromkeys(TASKS, 0.48))
            _summary(d, "cand", dict.fromkeys(TASKS, 0.90))
            # no baseline summary on disk
            v = ar_promote.run_gate("cand", "e5-base,lfm2.5", "v6-1-baseline-512", 0.005, mteb_root=d)
            self.assertFalse(v["promotable"])
            self.assertIn("baseline_present", v["failed_gates"])

    def test_missing_candidate_summary_is_gate_error(self):
        with tempfile.TemporaryDirectory() as d:
            _summary(d, "e5-base", dict.fromkeys(TASKS, 0.5))
            v = ar_promote.run_gate("ghost", "e5-base,lfm2.5", "v6-1-baseline", 0.005, mteb_root=d)
            self.assertFalse(v["promotable"])
            # a missing CANDIDATE summary is classified distinctly from a peer/baseline setup error
            self.assertEqual(v.get("error_kind"), "candidate_summary_missing")

    def test_missing_peer_summary_is_setup_error_not_candidate_failure(self):
        with tempfile.TemporaryDirectory() as d:
            # candidate + baseline present, but a peer summary is absent → setup error, not a
            # candidate failure misattribution (#8).
            _summary(d, "cand", dict.fromkeys(TASKS, 0.9))
            _summary(d, "v6-1-baseline", dict.fromkeys(TASKS, 0.4))
            v = ar_promote.run_gate("cand", "e5-base,lfm2.5", "v6-1-baseline", 0.005, mteb_root=d)
            self.assertFalse(v["promotable"])
            self.assertEqual(v.get("error_kind"), "setup_error_missing_peer_or_baseline_summary")
            self.assertIn("SETUP ERROR", ar_promote.render_report(v))

    def test_report_renders(self):
        with tempfile.TemporaryDirectory() as d:
            _fixture_root(d, [0.90, 0.90, 0.90, 0.90])
            v = ar_promote.run_gate("cand", "e5-base,lfm2.5", "v6-1-baseline-512", 0.005, mteb_root=d)
            md = ar_promote.render_report(v)
            self.assertIn("promotion report", md)
            self.assertIn("promotable", md)


class TestMtebTrial(unittest.TestCase):
    CFG = {"tasks": TASKS, "loader": "st", "batch_size": 32, "max_seq_length": 512,
           "long_doc_native_context": {"enabled": False}}

    def test_primary_command(self):
        plan = ar_mteb_trial.build_mteb_commands(self.CFG, "outputs/v8/x/checkpoint", "x")
        self.assertEqual(len(plan["commands"]), 1)
        self.assertIn("run_mteb_retrieval_de.py", plan["commands"][0])
        self.assertIn("--max-seq-length 512", plan["commands"][0])
        self.assertEqual(plan["summary_path"], "outputs/mteb/x/summary.json")

    def test_longdoc_second_pass(self):
        cfg = dict(self.CFG, long_doc_native_context={"enabled": True,
                   "tasks": ["GerDaLIRSmall"], "max_seq_length": 2048})
        plan = ar_mteb_trial.build_mteb_commands(cfg, "m", "x")
        self.assertEqual(len(plan["commands"]), 2)
        self.assertIn("2048", plan["commands"][1])

    def test_dry_run_main_writes_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "c.json"
            cfg.write_text(json.dumps(self.CFG), encoding="utf-8")
            rc = ar_mteb_trial.main(["--config", str(cfg), "--model", "m", "--label", "lbl",
                                     "--out", f"{d}/mteb", "--dry-run"])
            self.assertEqual(rc, 0)
            self.assertTrue((Path(d) / "mteb" / "lbl" / "mteb_trial_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
