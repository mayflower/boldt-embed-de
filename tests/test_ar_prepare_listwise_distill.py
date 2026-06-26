"""P09 — listwise-KL prepare/validate (stdlib): fail-closed list validation + teacher-list planning."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "ar_prepare_listwise_distill", ROOT / "scripts" / "ar_prepare_listwise_distill.py")
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def _valid_list():
    return {"query": "was ist x?", "positive_doc_ids": ["d1"],
            "candidates": [
                {"doc_id": "d1", "label": 1, "teacher_softmax_target": 0.7, "teacher_score": 5.0},
                {"doc_id": "d2", "label": 0, "teacher_softmax_target": 0.2, "teacher_score": 1.0},
                {"doc_id": "d3", "label": 0, "teacher_softmax_target": 0.1, "teacher_score": 0.5}]}


def _write(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


class TestValidate(unittest.TestCase):
    def test_no_torch(self):
        import pathlib as _p
        sys.path.insert(0, str(_p.Path(__file__).resolve().parent))
        from torch_free import script_is_torch_free
        self.assertTrue(script_is_torch_free("scripts/ar_prepare_listwise_distill.py"))

    def test_valid_lists_pass(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "lists.jsonl"
            _write(p, [_valid_list(), _valid_list()])
            stats, errors = prep.validate_listwise_file(p)
            self.assertEqual(errors, [], errors)
            self.assertEqual(stats["with_teacher"], 2)
            self.assertEqual(stats["with_positive"], 2)

    def test_missing_file_fails(self):
        stats, errors = prep.validate_listwise_file(Path("/nope/lists.jsonl"))
        self.assertTrue(any("not found" in e for e in errors))

    def test_too_few_candidates_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "lists.jsonl"
            r = _valid_list(); r["candidates"] = r["candidates"][:1]
            _write(p, [r])
            _, errors = prep.validate_listwise_file(p)
            self.assertTrue(any("< 2 candidates" in e for e in errors))

    def test_no_teacher_signal_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "lists.jsonl"
            r = _valid_list()
            for c in r["candidates"]:
                c.pop("teacher_softmax_target"); c.pop("teacher_score")
            _write(p, [r])
            _, errors = prep.validate_listwise_file(p)
            self.assertTrue(any("teacher signal" in e for e in errors))

    def test_no_positive_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "lists.jsonl"
            r = _valid_list(); r.pop("positive_doc_ids")
            for c in r["candidates"]:
                c["label"] = 0; c.pop("high_precision_positive", None)
            _write(p, [r])
            _, errors = prep.validate_listwise_file(p)
            self.assertTrue(any("positive" in e for e in errors))

    def test_eval_path_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "eval"; sub.mkdir()
            p = sub / "lists.jsonl"
            _write(p, [_valid_list()])
            _, errors = prep.validate_listwise_file(p)
            self.assertTrue(any("eval-derived" in e for e in errors))


class TestTeacherPlanAndMain(unittest.TestCase):
    def test_plan_new_teacher_lists(self):
        plan = prep.plan_new_teacher_lists(
            {"new_teacher_lists": {"enabled": True, "source_ids": ["swim_ir_de_full"], "slice_rows": 1000}})
        self.assertEqual(len(plan["planned_commands"]), 2)
        self.assertIn("build_v6_candidate_union.py", plan["planned_commands"][0])
        self.assertIn("score_rag_candidate_lists.py", plan["planned_commands"][1])

    def test_dry_run_existing_lists_ok(self):
        with tempfile.TemporaryDirectory() as d:
            lists = Path(d) / "lists.jsonl"; _write(lists, [_valid_list(), _valid_list()])
            cfg = Path(d) / "c.json"
            cfg.write_text(json.dumps({"name": "t", "lists": str(lists),
                                       "new_teacher_lists": {"enabled": False}}), encoding="utf-8")
            rc = prep.main(["--config", str(cfg), "--out", f"{d}/out", "--dry-run"])
            self.assertEqual(rc, 0)

    def test_dry_run_new_teacher_is_planned_not_run(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "c.json"
            cfg.write_text(json.dumps({"name": "t", "lists": "x",
                                       "new_teacher_lists": {"enabled": True,
                                                             "source_ids": ["swim_ir_de_full"]}}),
                           encoding="utf-8")
            rc = prep.main(["--config", str(cfg), "--out", f"{d}/out", "--dry-run"])
            self.assertEqual(rc, 0)  # planned, not executed


if __name__ == "__main__":
    unittest.main()
