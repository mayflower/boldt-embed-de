"""Tests for the advisory v7 EmbedFilter gate (pure stdlib)."""
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load("check_embedfilter_gate")
ACTIVE = ["webfaq_heldout", "germanquad", "dt_test"]


def _rows(*, d512_full=-0.002, r512_full=-0.001, d256_prefix=0.003, r256_prefix=0.002,
          guard_d=-0.001):
    rows = []
    for s in ACTIVE:
        rows.append({"method": "full", "dim": 1024, "eval_set": s, "role": "active",
                     "ndcg@10": 0.70, "recall@100": 0.97})
        # τ2 / 512 (deltas vs full); use guard_d for the guardrail sets
        dn = guard_d if s in ("germanquad", "dt_test") else d512_full
        rows.append({"method": "embedfilter", "dim": 512, "tau": 2, "eval_set": s, "role": "active",
                     "ndcg@10": 0.69, "recall@100": 0.965,
                     "dNDCG10_vs_full": dn, "dRecall100_vs_full": r512_full})
        # τ4 / 256 (deltas vs prefix-256 and vs full)
        rows.append({"method": "embedfilter", "dim": 256, "tau": 4, "eval_set": s, "role": "active",
                     "ndcg@10": 0.68, "recall@100": 0.96,
                     "dNDCG10_vs_full": guard_d if s in ("germanquad", "dt_test") else -0.004,
                     "dNDCG10_vs_prefix": d256_prefix, "dRecall100_vs_prefix": r256_prefix})
    return rows


class GateTests(unittest.TestCase):
    def test_pass_when_competitive(self):
        self.assertEqual(G.embedfilter_gate(_rows())["status"], "pass")

    def test_fail_when_512_far_below_full(self):
        r = G.embedfilter_gate(_rows(d512_full=-0.02))
        self.assertEqual(r["status"], "fail")
        self.assertIn("tau2_512_within_tol_of_full", r["failed"])

    def test_fail_when_256_below_prefix(self):
        r = G.embedfilter_gate(_rows(d256_prefix=-0.01))
        self.assertIn("tau4_256_matches_or_beats_prefix256", r["failed"])

    def test_fail_on_guardrail_regression(self):
        r = G.embedfilter_gate(_rows(guard_d=-0.02))
        self.assertIn("germanquad_dttest_guardrail", r["failed"])

    def test_diagnostic_rows_excluded(self):
        rows = _rows() + [{"method": "embedfilter", "dim": 512, "tau": 2, "eval_set": "gerdalir",
                           "role": "diagnostic", "ndcg@10": 0.01, "dNDCG10_vs_full": -0.5}]
        self.assertEqual(G.embedfilter_gate(rows)["status"], "pass")   # gerdalir ignored

    def test_require_real_fails_without_metrics(self):
        r = G.embedfilter_gate([], require_real=True)
        self.assertEqual(r["status"], "fail")

    def test_main_missing_sweep_require_real_returns_1(self):
        rc = G.main(["--sweep", "/nonexistent/sweep.json", "--require-real"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
