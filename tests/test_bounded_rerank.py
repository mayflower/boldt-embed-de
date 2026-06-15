"""Stdlib tests for inference-only bounded reranking policies. No ML, no labels at inference."""
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import bounded_rerank as BR  # noqa: E402

DEV = ROOT / "tests/fixtures/rag_dev_lists_scored.jsonl"
EVAL = ROOT / "tests/fixtures/rag_eval_lists_scored.jsonl"


def mklist(qid, pos_idx, fs_scores, rr_scores):
    cands = [{"doc_id": f"{qid}-d{i}", "text": f"doc {i}", "candidate_source": "bm25",
              "first_stage_rank": i, "first_stage_score": fs_scores[i], "reranker_score": rr_scores[i],
              "is_positive": i == pos_idx} for i in range(len(fs_scores))]
    return {"query_id": qid, "query": f"q-{qid}", "positive_doc_ids": [f"{qid}-d{pos_idx}"],
            "candidates": cands}


def _fs_ids(row):
    return [c["doc_id"] for c in BR._first_stage(row["candidates"])]


class TestLocks(unittest.TestCase):
    def test_top1_lock_never_moves_first_stage_rank1(self):
        row = mklist("a", 0, [20, 5, 4, 3, 2, 1, 0.9, 0.8, 0.7, 0.6],
                            [0.0, 9, 8, 7, 6, 5, 4, 3, 2, 1])     # reranker would demote top1
        out, action = BR.apply_policy(row, "top1_lock", {})
        self.assertEqual(out[0], _fs_ids(row)[0])
        self.assertEqual(action, "top1_lock")

    def test_topk_lock_preserves_topk_relative_order(self):
        row = mklist("b", 0, [20, 19, 18, 5, 4, 3, 2, 1, 0.5, 0.2],
                            [1, 5, 3, 9, 8, 7, 6, 4, 2, 0.5])
        for k in (1, 2, 3, 5):
            out, _ = BR.apply_policy(row, "topk_lock", {"k": k})
            self.assertEqual(out[:k], _fs_ids(row)[:k], f"k={k}")

    def test_bounded_downshift_respects_D(self):
        row = mklist("c", 0, [20, 18, 16, 14, 12, 10, 8, 6, 4, 2],
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 9.0])  # rr wants to upend
        fs = _fs_ids(row)
        fs_rank = {d: i for i, d in enumerate(fs)}
        for D in (1, 2, 3, 5):
            out, _ = BR.apply_policy(row, "bounded_downshift", {"D": D})
            for d in fs:
                self.assertLessEqual(out.index(d), fs_rank[d] + D, f"D={D} doc {d}")

    def test_margin_override_breaks_lock_only_when_margin_met(self):
        # reranker top beats first-stage top1 by 3.0
        row = mklist("d", 0, [20, 5, 4, 3, 2, 1, 0.9, 0.8, 0.7, 0.6],
                            [1.0, 4.0, 2, 1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        keep, _ = BR.apply_policy(row, "margin_override", {"margin": 5.0})   # 3.0 < 5 -> lock
        self.assertEqual(keep[0], _fs_ids(row)[0])
        brk, _ = BR.apply_policy(row, "margin_override", {"margin": 2.0})    # 3.0 >= 2 -> rerank
        self.assertEqual(brk[0], "d-d1")                                     # reranker top wins

    def test_blend_alpha1_equals_first_stage(self):
        row = mklist("e", 0, [20, 5, 4, 3, 2, 1, 0.9, 0.8, 0.7, 0.6],
                            [0.0, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        out, _ = BR.apply_policy(row, "blend", {"alpha": 1.0})
        self.assertEqual(out, _fs_ids(row))


class TestNoLabelsAtInference(unittest.TestCase):
    def test_policy_ignores_qrels_labels(self):
        row = json.loads(pathlib.Path(DEV).read_text("utf-8").split("\n")[0])
        stripped = json.loads(json.dumps(row))
        stripped.pop("positive_doc_ids", None)
        for c in stripped["candidates"]:
            c.pop("is_positive", None); c.pop("label", None); c.pop("teacher_score", None)
        for pol in ("always_rerank", "top1_lock", "topk_lock", "bounded_downshift",
                    "margin_override", "blend", "confidence_conditional", "combined_safe_policy"):
            params = {"k": 3, "D": 3, "U": 3, "alpha": 0.5, "margin": 2.0, "fs_gap_high": 1.0,
                      "fs_gap_med": 0.1}
            self.assertEqual(BR.apply_policy(row, pol, params)[0],
                             BR.apply_policy(stripped, pol, params)[0],
                             f"{pol} differs when labels stripped -> uses labels at inference")


class TestDeterminism(unittest.TestCase):
    def test_order_independent_and_repeatable(self):
        row = mklist("z", 0, [20, 5, 4, 3, 2, 1, 0.9, 0.8, 0.7, 0.6],
                            [0.0, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        rev = json.loads(json.dumps(row)); rev["candidates"] = list(reversed(rev["candidates"]))
        for pol in ("always_rerank", "top1_lock", "bounded_downshift", "blend"):
            a, _ = BR.apply_policy(row, pol, {"D": 2, "alpha": 0.5})
            b, _ = BR.apply_policy(rev, pol, {"D": 2, "alpha": 0.5})
            self.assertEqual(a, b, pol)


class TestCatastrophicPrevention(unittest.TestCase):
    def test_top1_lock_prevents_always_rerank_catastrophe(self):
        # correct positive at first-stage rank0; reranker sends it to the bottom
        row = mklist("k", 0, [20, 5, 4, 3, 2, 1, 0.9, 0.8, 0.7, 0.6],
                            [0.0, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        always = BR.evaluate_policy([row], "always_rerank", {})
        lock = BR.evaluate_policy([row], "top1_lock", {})
        self.assertEqual(always["catastrophic_drop_rate"], 1.0)
        self.assertEqual(lock["catastrophic_drop_rate"], 0.0)
        self.assertGreater(lock["policy_ndcg@10"], always["policy_ndcg@10"])


class TestFit(unittest.TestCase):
    def test_grid_search_dev_only(self):
        dev = [json.loads(l) for l in pathlib.Path(DEV).read_text("utf-8").split("\n") if l.strip()]
        fit = BR.grid_search(dev)
        self.assertEqual(fit["fit_on"], "dev_only")
        self.assertIn(fit["policy"], BR.POLICIES)

    def test_fit_on_dev_eval_on_guardrail_without_guardrail_labels(self):
        dev = [json.loads(l) for l in pathlib.Path(DEV).read_text("utf-8").split("\n") if l.strip()]
        fit = BR.grid_search(dev)
        # "guardrail" eval set — strip its labels to PROVE the policy decision needs none
        guard = [json.loads(l) for l in pathlib.Path(EVAL).read_text("utf-8").split("\n") if l.strip()]
        stripped = json.loads(json.dumps(guard))
        for r in stripped:
            r.pop("positive_doc_ids", None)
            for c in r["candidates"]:
                c.pop("is_positive", None)
        for r_full, r_strip in zip(guard, stripped):
            self.assertEqual(BR.apply_policy(r_full, fit["policy"], fit["best_params"]),
                             BR.apply_policy(r_strip, fit["policy"], fit["best_params"]))

    def test_eval_report_has_required_fields(self):
        ev = [json.loads(l) for l in pathlib.Path(EVAL).read_text("utf-8").split("\n") if l.strip()]
        rep = BR.evaluate_policy(ev, "top1_lock", {})
        for k in ("policy", "abstain_rate", "lock_rate", "avg_max_displacement",
                  "first_stage_ndcg@10", "policy_ndcg@10", "delta_vs_first_stage",
                  "catastrophic_drop_rate", "by_bucket", "top_catastrophic_examples"):
            self.assertIn(k, rep)


if __name__ == "__main__":
    unittest.main()
