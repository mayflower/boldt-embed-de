"""Tests for the conservative reranker's rank-preservation loss + high-confidence detection."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import rank_preservation_loss as RP  # noqa: E402  (module import is torch-free)


def row(fs_scores, rr_scores, teacher_scores, pos_idx=0):
    cands = [{"doc_id": f"d{i}", "text": f"doc {i}", "candidate_source": "bm25",
              "first_stage_rank": i, "first_stage_score": fs_scores[i],
              "reranker_score": rr_scores[i], "teacher_score": teacher_scores[i],
              "teacher_softmax_target": None, "is_positive": i == pos_idx} for i in range(len(fs_scores))]
    return {"query_id": "q", "query": "q?", "positive_doc_ids": [f"d{pos_idx}"], "candidates": cands}


class TestLoss(unittest.TestCase):
    def test_zero_when_order_preserved(self):
        import torch
        loss = RP.rank_preservation_loss(torch.tensor([4.0, 1.0]), [0, 1], torch.tensor([1.0, 1.0]))
        self.assertAlmostEqual(float(loss), 0.0, 5)

    def test_positive_when_top1_moved_without_margin(self):
        import torch
        # student flips order (doc1 above doc0); teacher gives no advantage -> penalized
        loss = RP.rank_preservation_loss(torch.tensor([1.0, 4.0]), [0, 1], torch.tensor([1.0, 1.0]),
                                         justify_margin=2.0)
        self.assertGreater(float(loss), 0.0)

    def test_teacher_margin_allows_justified_movement(self):
        import torch
        # same flip, but teacher strongly prefers doc1 (margin 4 >= 2) -> allowed, no penalty
        loss = RP.rank_preservation_loss(torch.tensor([1.0, 4.0]), [0, 1], torch.tensor([1.0, 5.0]),
                                         justify_margin=2.0)
        self.assertAlmostEqual(float(loss), 0.0, 5)

    def test_moving_top1_down_costs_more_than_a_small_swap(self):
        import torch
        ranks = [0, 1, 2, 3]
        teacher = torch.tensor([1.0, 1.0, 1.0, 1.0])      # no justification anywhere
        move_top1 = RP.rank_preservation_loss(torch.tensor([0.0, 3.0, 2.0, 1.0]), ranks, teacher)
        small_swap = RP.rank_preservation_loss(torch.tensor([4.0, 3.0, 1.0, 2.0]), ranks, teacher)
        self.assertGreater(float(move_top1), float(small_swap))

    def test_loss_is_differentiable(self):
        import torch
        s = torch.tensor([1.0, 4.0], requires_grad=True)
        loss = RP.rank_preservation_loss(s, [0, 1], torch.tensor([1.0, 1.0]))
        loss.backward()
        self.assertIsNotNone(s.grad)


class TestHighConfidence(unittest.TestCase):
    def test_detection_uses_no_qrels(self):
        r = row([20.0, 5.0, 4.0, 3.0], [1, 2, 3, 4], [6, 1, 1, 1])     # big first-stage gap
        self.assertTrue(RP.is_high_confidence(r, fs_gap_min=5.0))
        stripped = json.loads(json.dumps(r))
        stripped.pop("positive_doc_ids", None)
        for c in stripped["candidates"]:
            c.pop("is_positive", None); c.pop("teacher_score", None); c.pop("label", None)
        self.assertEqual(RP.is_high_confidence(r, fs_gap_min=5.0),
                         RP.is_high_confidence(stripped, fs_gap_min=5.0))

    def test_low_gap_not_high_confidence(self):
        r = row([6.0, 5.8, 5.6, 5.4], [1, 2, 3, 4], [6, 1, 1, 1])
        self.assertFalse(RP.is_high_confidence(r, fs_gap_min=5.0))

    def test_conservative_batches_carry_flags(self):
        rows = [row([20.0, 5.0, 4.0, 3.0], [1, 2, 3, 4], [6, 1, 1, 1]),
                row([6.0, 5.8, 5.6, 5.4], [1, 2, 3, 4], [6, 1, 1, 1])]
        batches = RP.scored_lists_to_conservative_batches(rows, fs_gap_min=5.0)
        self.assertEqual(len(batches), 2)
        for b in batches:
            self.assertIn("high_confidence", b)
            self.assertIn("first_stage_ranks", b)
            self.assertIn("teacher_scores", b)
            self.assertEqual(len(b["target"]), 4)


class TestPlanAndDryRun(unittest.TestCase):
    def test_plan_includes_preservation(self):
        plan = RP.plan_conservative_loss(0.2)
        self.assertIn("rank_preservation_loss(lambda=0.2)", " ".join(plan["components"]))
        self.assertEqual(plan["weights"]["preservation"], 0.2)

    def test_cli_dry_run_no_torch(self):
        with tempfile.TemporaryDirectory() as d:
            lists = pathlib.Path(d) / "scored.jsonl"
            rows = [row([20.0, 5.0, 4.0, 3.0], [1, 2, 3, 4], [6, 1, 1, 1]),
                    row([6.0, 5.8, 5.6, 5.4], [1, 2, 3, 4], [6, 1, 1, 1])]
            # give them distinct domains so the FAQ cap / not-faq-only check passes
            rows[0]["domain"] = "qa_passage_non_eval"; rows[1]["domain"] = "web_nonfaq"
            lists.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            report = pathlib.Path(d) / "card.json"
            code = (
                "import sys; sys.path.insert(0, %r); "
                "sys.argv=['x','--config', %r,'--candidate-lists', %r,'--report', %r,"
                "'--run-id','t','--max-faq-share','1.0','--dry-run']\n"
                "import runpy; rc=0\n"
                "try:\n runpy.run_path(%r, run_name='__main__')\n"
                "except SystemExit as e:\n rc=e.code or 0\n"
                "assert 'torch' not in sys.modules, 'torch imported'\n"
                "print('RC', rc)\n"
                % (str(ROOT / "src"),
                   str(ROOT / "configs" / "experiments" / "v5_small_rag.json"),
                   str(lists), str(report),
                   str(ROOT / "scripts" / "train_v5_rag_reranker_conservative.py"))
            )
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("RC 0", r.stdout)


if __name__ == "__main__":
    unittest.main()
