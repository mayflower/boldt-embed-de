"""P06 — dense trial generalization: plan + real command surface grad-accum / mini-batch / seq cap.

The recipe must expose effective_batch_size = batch × grad_accumulation, the GradCache mini-batch,
gradient_checkpointing and max_triplets_per_query, forward exactly the trainer-supported flags into
the real command, keep the memory seq cap visible, and never claim an inert knob is active."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import autoresearch_recipe as r  # noqa: E402


def _cfg(**training):
    base = {"batch_size": 32, "max_query_length": 256, "max_document_length": 512}
    base.update(training)
    return {"task": "dense_retriever", "training": base}


class TestPlanFields(unittest.TestCase):
    def test_no_torch(self):
        import pathlib as _p
        sys.path.insert(0, str(_p.Path(__file__).resolve().parent))
        from torch_free import module_is_torch_free
        self.assertTrue(module_is_torch_free("boldt_embed.autoresearch_recipe"))

    def test_effective_batch_is_batch_times_accum(self):
        plan = r.build_training_plan(_cfg(batch_size=32, grad_accumulation=8))
        self.assertEqual(plan["grad_accumulation"], 8)
        self.assertEqual(plan["effective_batch_size"], 256)

    def test_new_fields_present(self):
        plan = r.build_training_plan(_cfg(mini_batch_size=8, gradient_checkpointing=True,
                                          max_triplets_per_query=5, temperature_schedule="constant"))
        for k in ("mini_batch_size", "effective_batch_size", "gradient_checkpointing",
                  "temperature_schedule", "max_triplets_per_query", "plan_only_knobs"):
            self.assertIn(k, plan)
        self.assertEqual(plan["mini_batch_size"], 8)
        self.assertTrue(plan["gradient_checkpointing"])

    def test_memory_cap_stays_visible(self):
        # batch 32 × requested doc 1024 would OOM -> seq capped, and the cap is recorded, not silent
        plan = r.build_training_plan(_cfg(batch_size=32, max_document_length=1024))
        self.assertTrue(plan["seq_capped_for_memory"])
        self.assertEqual(plan["max_seq_length_requested"], 1024)
        self.assertLess(plan["max_seq_length"], 1024)

    def test_nonconstant_temp_schedule_marked_plan_only(self):
        plan = r.build_training_plan(_cfg(temperature_schedule="cosine"))
        self.assertIn("temperature_schedule", plan["plan_only_knobs"])

    def test_constant_temp_schedule_not_plan_only(self):
        plan = r.build_training_plan(_cfg(temperature_schedule="constant"))
        self.assertEqual(plan["plan_only_knobs"], [])

    def test_inconsistent_effective_batch_flagged(self):
        plan = r.build_training_plan(_cfg(batch_size=32, grad_accumulation=2,
                                          effective_batch_size=999))
        self.assertTrue(any("effective_batch_size" in k for k in plan["plan_only_knobs"]))


class TestRealCommand(unittest.TestCase):
    def _cmd(self, **training):
        plan = r.build_training_plan(_cfg(**training))
        return r._build_train_cmd(Path("scripts/train_v6_1_dense_top50.py"),
                                  train_base="base", train_pairs="pairs.jsonl",
                                  hard_negs="hn.jsonl", ckpt=Path("/tmp/ckpt"),
                                  steps=100, plan=plan, run_id="t1")

    def test_grad_accum_forwarded_when_gt1(self):
        cmd = self._cmd(batch_size=32, grad_accumulation=8)
        self.assertIn("--grad-accumulation", cmd)
        self.assertIn("8", cmd)

    def test_grad_accum_not_forwarded_when_1(self):
        self.assertNotIn("--grad-accumulation", self._cmd(grad_accumulation=1))

    def test_mini_batch_and_ckpt_and_triplets_forwarded(self):
        cmd = self._cmd(mini_batch_size=8, gradient_checkpointing=True, max_triplets_per_query=5)
        self.assertIn("--mini-batch-size", cmd)
        self.assertIn("--gradient-checkpointing", cmd)
        self.assertIn("--max-triplets-per-query", cmd)


if __name__ == "__main__":
    unittest.main()
