"""Tests for the v6 RAW reranker lift eval + promotion gate (stdlib, no ML)."""
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GATE = _load("check_v6_raw_reranker_gate")
EVAL = _load("eval_v6_raw_reranker_lift")


def _rep(name, role, *, delta, catastrophic=0.0, present=1.0, mode="raw"):
    return {"eval_set": name, "role": role, "ranking_mode": mode, "delta_ndcg@10": delta,
            "catastrophic_drop_rate": catastrophic, "positive_present_rate": present}


def _passing():
    return {
        "webfaq": _rep("webfaq", "primary", delta=0.12, present=0.95),
        "germanquad": _rep("germanquad", "guardrail", delta=-0.001, catastrophic=0.01),
        "dt_test": _rep("dt_test", "guardrail", delta=0.0, catastrophic=0.0),
    }


class GateTests(unittest.TestCase):
    def test_passes_clean_fixture(self):
        g = GATE.raw_reranker_gate(_passing())
        self.assertEqual(g["status"], "pass", g["failing"])
        self.assertFalse(g["policy_gated_result_used"])

    def test_rejects_policy_result(self):
        for mode in ("bounded_margin_override", "policy_gated", "abstain"):
            r = _passing()
            r["webfaq"]["ranking_mode"] = mode
            g = GATE.raw_reranker_gate(r)
            self.assertEqual(g["status"], "fail", mode)
            self.assertTrue(any(c["check"].startswith("raw_only") for c in g["failing"]), mode)

    def test_fails_on_germanquad_negative_delta(self):
        r = _passing()
        r["germanquad"]["delta_ndcg@10"] = -0.03   # below -0.005
        g = GATE.raw_reranker_gate(r)
        self.assertEqual(g["status"], "fail")
        self.assertIn("germanquad_delta", [c["check"] for c in g["failing"]])

    def test_fails_on_catastrophic_rate(self):
        r = _passing()
        r["germanquad"]["catastrophic_drop_rate"] = 0.10   # above 0.03
        g = GATE.raw_reranker_gate(r)
        self.assertEqual(g["status"], "fail")
        self.assertIn("germanquad_catastrophic", [c["check"] for c in g["failing"]])

    def test_ignores_gerdalir_diagnostic(self):
        r = _passing()
        r["gerdalir"] = _rep("gerdalir", "diagnostic", delta=-0.9, catastrophic=0.9, mode="bounded")
        g = GATE.raw_reranker_gate(r)
        self.assertEqual(g["status"], "pass", g["failing"])
        self.assertIn("gerdalir", g["ignored_diagnostic_sets"])

    def test_fails_on_low_positive_present_rate(self):
        r = _passing()
        r["webfaq"]["positive_present_rate"] = 0.5   # eval not meaningful
        g = GATE.raw_reranker_gate(r)
        self.assertEqual(g["status"], "fail")
        self.assertIn("webfaq_positive_present", [c["check"] for c in g["failing"]])

    def test_fails_on_leakage_and_policy_card(self):
        self.assertEqual(GATE.raw_reranker_gate(_passing(), leakage=True)["status"], "fail")
        self.assertEqual(
            GATE.raw_reranker_gate(_passing(), card_recommends_policy=True)["status"], "fail")

    def test_webfaq_below_bar_fails(self):
        r = _passing()
        r["webfaq"]["delta_ndcg@10"] = 0.02   # < +0.05
        g = GATE.raw_reranker_gate(r)
        self.assertEqual(g["status"], "fail")
        self.assertIn("webfaq_delta", [c["check"] for c in g["failing"]])


class EvalCoreTests(unittest.TestCase):
    def _row(self, qid, gold_fs_rank, n=6):
        cands = [{"doc_id": f"{qid}-d{i}", "text": f"t{i}", "first_stage_rank": i,
                  "label": 1 if i == gold_fs_rank else 0} for i in range(n)]
        return {"query_id": qid, "query": f"q{qid}", "positive_doc_ids": [f"{qid}-d{gold_fs_rank}"],
                "candidates": cands}

    def test_raw_mode_and_lift(self):
        # gold at first-stage rank 4; reranker scores promote it to the top -> positive lift
        row = self._row("q1", 4)
        scores = {"q1": [0.1, 0.2, 0.3, 0.4, 9.0, 0.0]}  # index 4 (gold) highest
        rep = EVAL.raw_lift_report([row], scores, "webfaq")
        self.assertEqual(rep["ranking_mode"], "raw")
        self.assertGreater(rep["raw_reranker_ndcg@10"], rep["first_stage_ndcg@10"])
        self.assertGreater(rep["delta_ndcg@10"], 0)
        self.assertTrue(rep["candidate_set_unchanged"])
        self.assertEqual(rep["positive_present_rate"], 1.0)

    def test_no_scores_is_noop(self):
        row = self._row("q1", 2)
        rep = EVAL.raw_lift_report([row], {}, "webfaq")   # no scores -> raw == first stage
        self.assertEqual(rep["delta_ndcg@10"], 0.0)
        self.assertTrue(rep["candidate_set_unchanged"])

    def test_catastrophic_counted(self):
        # gold at rank 0 (perfect first stage); reranker buries it -> catastrophic drop
        row = self._row("q1", 0)
        scores = {"q1": [-9.0, 1.0, 1.0, 1.0, 1.0, 1.0]}  # gold (idx0) pushed to bottom
        rep = EVAL.raw_lift_report([row], scores, "germanquad")
        self.assertEqual(rep["catastrophic_drop_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
