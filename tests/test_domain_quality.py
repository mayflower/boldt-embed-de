"""Tests for v3 domain-quality gates (pure stdlib). The gate must catch v2's failure mode:
nominally multi-domain but effectively web/wiki, or real domains the teacher rejects."""
import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import domain_quality as dq  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
SCRIPT = ROOT / "scripts" / "analyze_domain_quality.py"

# small floors so tiny fixtures fail on the SPECIFIC gate under test, not the 5000 default.
SMALL = {"min_real_domain_accepted": {"faq_real": 1, "admin_real": 1,
                                      "legal_adjacency_real_no_eval_overlap": 1},
         "max_synthetic_share_for_real_domains": 0.25,
         "max_effective_web_wiki_share": 0.65, "min_teacher_acceptance_rate": 0.35}


def _row(dom, i, *, score=6.5, lic="CC-BY-4.0", synthetic=False, positive=True):
    rid = f"{dom}_{i}"
    r = {"id": rid, "query_id": rid, "doc_id": rid, "query": f"q{i}",
         "document": f"doc {dom} {i}", "domain": dom, "source": f"{dom}_src",
         "license": lic, "license_origin": "inherited" if synthetic else "manifest",
         "positive": positive, "embedding_score": 0.5, "reranker_score": score}
    return r


def _failing(report, gate, domain=None):
    return any(g["gate"] == gate and (domain is None or g["domain"] == domain)
               for g in report["failing_gates"])


class TestGates(unittest.TestCase):
    def test_low_real_domain_acceptance_fails(self):
        # faq_real: 10 raw, only 1 accepted (teacher rejects the synthetic rest) -> v2 failure mode
        cands = ([_row("faq_real", i) for i in range(10)]
                 + [_row("admin_real", 0)] + [_row("legal_adjacency_real_no_eval_overlap", 0)])
        cache = ([_row("faq_real", 0)] + [_row("faq_real", i, score=0.1, synthetic=True)
                                          for i in range(1, 10)]
                 + [_row("admin_real", 0)] + [_row("legal_adjacency_real_no_eval_overlap", 0)])
        rep = dq.analyze(cands, cache, gates=SMALL)
        self.assertEqual(rep["status"], "fail")
        self.assertTrue(_failing(rep, "real_domain_acceptance_rate", "faq_real"))
        self.assertAlmostEqual(rep["per_domain"]["faq_real"]["acceptance_rate"], 0.1)

    def test_high_synthetic_share_fails(self):
        cache = ([_row("faq_real", 0)]                                   # 1 real accepted
                 + [_row("faq_real", i, synthetic=True) for i in range(1, 4)]  # 3 synthetic accepted
                 + [_row("admin_real", 0)] + [_row("legal_adjacency_real_no_eval_overlap", 0)])
        rep = dq.analyze(cache, cache, gates=SMALL)
        self.assertTrue(_failing(rep, "real_domain_synthetic_share", "faq_real"))
        self.assertEqual(rep["per_domain"]["faq_real"]["synthetic_share"], 0.75)

    def test_all_unknown_licenses_fails(self):
        cache = [_row("faq_real", i, lic="unknown") for i in range(3)]
        rep = dq.analyze(cache, cache, gates=SMALL)
        self.assertTrue(_failing(rep, "license_unknown_rows_zero"))
        self.assertEqual(rep["totals"]["license_unknown_rows"], 3)

    def test_web_wiki_dominance_fails(self):
        cache = ([_row("web", i) for i in range(8)] + [_row("wiki_non_eval", i) for i in range(8)]
                 + [_row("faq_real", 0)] + [_row("admin_real", 0)]
                 + [_row("legal_adjacency_real_no_eval_overlap", 0)])
        rep = dq.analyze(cache, cache, gates=SMALL)
        self.assertTrue(_failing(rep, "effective_web_wiki_share"))
        self.assertGreater(rep["totals"]["effective_web_wiki_share"], 0.65)

    def test_legal_below_floor_blocks_claim(self):
        cache = [_row("faq_real", 0), _row("admin_real", 0)]   # no legal at all
        rep = dq.analyze(cache, cache, gates=SMALL)
        self.assertFalse(rep["can_claim_legal_transfer_from_data"])

    def test_healthy_fixture_passes(self):
        cands = [json.loads(l) for l in (FIX / "candidates_domain_quality.jsonl").read_text("utf-8").splitlines()]
        cache = [json.loads(l) for l in (FIX / "teacher_cache_domain_quality.jsonl").read_text("utf-8").splitlines()]
        gates = json.loads((FIX / "v3_real_domain_generalization.json").read_text("utf-8"))["domain_quality_gates"]
        rep = dq.analyze(cands, cache, gates=gates)
        self.assertEqual(rep["status"], "pass", rep["failing_gates"])
        self.assertTrue(rep["can_claim_legal_transfer_from_data"])
        self.assertLessEqual(rep["totals"]["effective_web_wiki_share"], 0.65)


class TestMarkdown(unittest.TestCase):
    def test_markdown_lists_failing_domains(self):
        cache = ([_row("faq_real", 0)] + [_row("faq_real", i, synthetic=True) for i in range(1, 4)]
                 + [_row("admin_real", 0)] + [_row("legal_adjacency_real_no_eval_overlap", 0)])
        md = dq.render_markdown(dq.analyze(cache, cache, gates=SMALL))
        self.assertIn("FAIL", md)
        self.assertIn("faq_real", md)
        self.assertIn("real_domain_synthetic_share", md)


class TestCli(unittest.TestCase):
    def test_cli_healthy_passes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            out, md = (pathlib.Path(d) / "q.json", pathlib.Path(d) / "q.md")
            r = subprocess.run(
                [sys.executable, str(SCRIPT),
                 "--candidates", str(FIX / "candidates_domain_quality.jsonl"),
                 "--teacher-cache", str(FIX / "teacher_cache_domain_quality.jsonl"),
                 "--config", str(FIX / "v3_real_domain_generalization.json"),
                 "--output", str(out), "--markdown", str(md)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.loads(out.read_text("utf-8"))["status"], "pass")
            self.assertIn("domain-quality report", md.read_text("utf-8"))

    def test_cli_blocks_on_failure(self):
        # point at a cache with unknown licenses via a temp file -> exit 1 (training blocked)
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cand = pathlib.Path(d) / "c.jsonl"
            cand.write_text("\n".join(json.dumps(_row("faq_real", i, lic="unknown"))
                                      for i in range(3)) + "\n", encoding="utf-8")
            r = subprocess.run(
                [sys.executable, str(SCRIPT), "--candidates", str(cand),
                 "--teacher-cache", str(cand), "--output", str(pathlib.Path(d) / "q.json"),
                 "--markdown", str(pathlib.Path(d) / "q.md")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)

    def test_module_is_stdlib_only(self):
        code = ("import sys; sys.path.insert(0, %r);"
                "from boldt_embed import domain_quality;"
                "assert 'torch' not in sys.modules; print('clean')") % str(ROOT / "src")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


if __name__ == "__main__":
    unittest.main()
