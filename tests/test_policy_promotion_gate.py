"""Tests for the frozen bounded-policy promotion gate (stdlib, no ML)."""
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GATE = _load_script("check_policy_promotion_gate")


def _rep(name, role, *, policy_delta, raw_delta, catastrophic, mode="policy_gated"):
    return {"eval_set": name, "role": role, "ranking_mode": mode, "policy_delta": policy_delta,
            "raw_delta": raw_delta, "catastrophic_drop_rate": catastrophic}


def _passing_reports():
    # WebFAQ lifts; the three near-ceiling/guardrail sets preserve and beat raw (which churns).
    return {
        "webfaq": _rep("webfaq", "primary", policy_delta=0.12, raw_delta=0.10, catastrophic=0.0),
        "near_ceiling": _rep("near_ceiling", "primary", policy_delta=0.0, raw_delta=-0.30,
                             catastrophic=0.0),
        "germanquad": _rep("germanquad", "guardrail", policy_delta=-0.002, raw_delta=-0.25,
                           catastrophic=0.01),
        "dt_test": _rep("dt_test", "guardrail", policy_delta=0.0, raw_delta=-0.20, catastrophic=0.0),
    }


class PromotionGateTests(unittest.TestCase):
    def test_gate_passes_complete_fixture(self):
        g = GATE.promotion_gate(_passing_reports())
        self.assertEqual(g["status"], "pass", g["failing"])
        self.assertFalse(g["raw_always_rerank_recommended"])

    def test_fails_if_near_ceiling_catastrophic_too_high(self):
        r = _passing_reports()
        r["near_ceiling"]["catastrophic_drop_rate"] = 0.05  # > 0.03
        g = GATE.promotion_gate(r)
        self.assertEqual(g["status"], "fail")
        self.assertIn("near_ceiling_catastrophic", [c["check"] for c in g["failing"]])

    def test_fails_if_tuned_on_guardrail_marker(self):
        g = GATE.promotion_gate(_passing_reports(), tuned_on_guardrail=True)
        self.assertEqual(g["status"], "fail")
        self.assertIn("not_tuned_on_guardrail", [c["check"] for c in g["failing"]])

    def test_ignores_gerdalir(self):
        r = _passing_reports()
        # GerDaLIR is diagnostic-only: even a catastrophic raw-mode GerDaLIR must not affect the gate.
        r["gerdalir"] = _rep("gerdalir", "diagnostic", policy_delta=-0.9, raw_delta=-0.9,
                             catastrophic=0.9, mode="raw_rerank")
        g = GATE.promotion_gate(r)
        self.assertEqual(g["status"], "pass", g["failing"])
        self.assertIn("gerdalir", g["ignored_diagnostic_sets"])

    def test_raw_rerank_cannot_pass_as_policy(self):
        r = _passing_reports()
        r["germanquad"]["ranking_mode"] = "raw_rerank"  # evaluated as raw always-rerank
        g = GATE.promotion_gate(r)
        self.assertEqual(g["status"], "fail")
        self.assertIn("not_raw_rerank", [c["check"] for c in g["failing"]])

    def test_fails_if_policy_loses_to_raw_on_germanquad(self):
        r = _passing_reports()
        r["germanquad"]["raw_delta"] = 0.10  # raw beats policy -> policy is not the safe choice
        g = GATE.promotion_gate(r)
        self.assertEqual(g["status"], "fail")
        self.assertIn("germanquad_beats_raw_rerank", [c["check"] for c in g["failing"]])

    def test_webfaq_below_threshold_fails(self):
        r = _passing_reports()
        r["webfaq"]["policy_delta"] = 0.02  # < 0.05
        g = GATE.promotion_gate(r)
        self.assertEqual(g["status"], "fail")
        self.assertIn("webfaq_policy_delta", [c["check"] for c in g["failing"]])


if __name__ == "__main__":
    unittest.main()
