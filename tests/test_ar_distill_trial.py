"""P09 — listwise-KL distill trial (stdlib): base/list validation, train command + MTEB eval plan."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "ar_distill_trial", ROOT / "scripts" / "ar_distill_trial.py")
dt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dt)


def _valid_list():
    return {"query": "q", "positive_doc_ids": ["d1"],
            "candidates": [{"doc_id": "d1", "label": 1, "teacher_softmax_target": 0.7},
                           {"doc_id": "d2", "label": 0, "teacher_softmax_target": 0.3}]}


class TestPlanDistill(unittest.TestCase):
    def test_no_torch(self):
        import pathlib as _p
        sys.path.insert(0, str(_p.Path(__file__).resolve().parent))
        from torch_free import script_is_torch_free
        self.assertTrue(script_is_torch_free("scripts/ar_distill_trial.py"))

    def test_valid_plan(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "base"; base.mkdir()
            lists = Path(d) / "lists.jsonl"
            lists.write_text("\n".join(json.dumps(_valid_list()) for _ in range(3)), encoding="utf-8")
            cfg = {"name": "v8_lwkl", "base_checkpoint": str(base), "lists": str(lists),
                   "output": f"{d}/out/checkpoint",
                   "training": {"steps": 1500, "contrastive_weight": 0.0, "tau": 0.05}}
            plan = dt.plan_distill(cfg)
            self.assertEqual(plan["errors"], [], plan["errors"])
            self.assertIn("train_listwise_kl.py", plan["command"])
            self.assertIn("--contrastive-weight", plan["command"])
            self.assertIn("0.0", plan["command"])
            self.assertEqual(len(plan["eval_plan"]), 2)
            self.assertIn("run_mteb_retrieval_de.py", plan["eval_plan"][0])
            self.assertIn("ar_promote.py", plan["eval_plan"][1])

    def test_missing_base_fails(self):
        with tempfile.TemporaryDirectory() as d:
            lists = Path(d) / "lists.jsonl"
            lists.write_text(json.dumps(_valid_list()), encoding="utf-8")
            plan = dt.plan_distill({"name": "x", "base_checkpoint": "outputs/nope/checkpoint",
                                    "lists": str(lists), "training": {}})
            self.assertTrue(any("base_checkpoint does not exist" in e for e in plan["errors"]))

    def test_invalid_lists_fail(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "base"; base.mkdir()
            plan = dt.plan_distill({"name": "x", "base_checkpoint": str(base),
                                    "lists": "/nope/lists.jsonl", "training": {}})
            self.assertTrue(any("not found" in e for e in plan["errors"]))

    def test_dry_run_main_writes_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "base"; base.mkdir()
            lists = Path(d) / "lists.jsonl"
            lists.write_text("\n".join(json.dumps(_valid_list()) for _ in range(3)), encoding="utf-8")
            cfg = Path(d) / "c.json"
            cfg.write_text(json.dumps({"name": "v8_lwkl", "base_checkpoint": str(base),
                                       "lists": str(lists), "output": f"{d}/o/checkpoint",
                                       "training": {"steps": 1000}}), encoding="utf-8")
            rc = dt.main(["--config", str(cfg), "--out", f"{d}/out", "--dry-run"])
            self.assertEqual(rc, 0)
            self.assertTrue((Path(d) / "out" / "v8_lwkl_distill_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
