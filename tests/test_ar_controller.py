"""P03 — ar_controller CLI: plan/status/next/record (stdlib, no GPU, never executes real runs)."""
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("ar_controller", ROOT / "scripts" / "ar_controller.py")
ar_controller = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ar_controller)

from boldt_embed import autoresearch_state as st  # noqa: E402

SEARCH_SPACE = str(ROOT / "configs" / "autoresearch" / "search_space_v8.json")


def _run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = ar_controller.main(argv)
    return rc, json.loads(buf.getvalue())


class TestController(unittest.TestCase):
    def test_no_torch(self):
        import pathlib as _p
        sys.path.insert(0, str(_p.Path(__file__).resolve().parent))
        from torch_free import script_is_torch_free
        self.assertTrue(script_is_torch_free("scripts/ar_controller.py"))

    def test_plan_dense_dry_run(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out = _run(["plan", "--trial-type", "dense", "--search-space", SEARCH_SPACE,
                            "--out", d, "--dry-run"])
            self.assertEqual(rc, 0)
            self.assertEqual(out["trial_type"], "dense")
            self.assertIn("ar_loop.py", out["command"])
            self.assertIn("--dry-run", out["command"])
            self.assertEqual(out["mode"], "dry-run")
            self.assertTrue((Path(d) / "plan.json").exists())

    def test_plan_real_flags_echoed_not_executed(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out = _run(["plan", "--trial-type", "dense", "--search-space", SEARCH_SPACE,
                            "--out", d, "--real", "--allow-gpu", "--allow-checkpoints"])
            self.assertEqual(rc, 0)
            self.assertIn("--real", out["command"])
            self.assertIn("--allow-gpu", out["command"])
            self.assertNotIn("--dry-run", out["command"])
            self.assertIn("planned only", out["mode"])

    def test_next_on_empty_state_plans_data_mix(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / "events.jsonl"
            rc, out = _run(["next", "--state", str(state), "--search-space", SEARCH_SPACE,
                            "--out", d, "--dry-run"])
            self.assertEqual(rc, 0)
            self.assertEqual(out["trial_type"], "data_mix")
            self.assertIn("ar_build_mixture.py", out["command"])
            self.assertIn("success_counts", out)

    def test_status_reports_next(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / "events.jsonl"
            st.append_event(st.new_event("data_mix", "ok", event_id="x1",
                                         timestamp_utc="2026-06-25T00:00:00+00:00",
                                         git={"commit": "c", "dirty": False}), state)
            rc, out = _run(["status", "--state", str(state), "--search-space", SEARCH_SPACE])
            self.assertEqual(rc, 0)
            self.assertEqual(out["n_events"], 1)
            self.assertEqual(out["next_trial_type"], "dense")

    def test_record_appends_event(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / "events.jsonl"
            ev = Path(d) / "ev.json"
            ev.write_text(json.dumps({"trial_type": "merge", "status": "ok",
                                      "notes": "test merge"}), encoding="utf-8")
            rc, out = _run(["record", "--event-json", str(ev), "--state", str(state),
                            "--search-space", SEARCH_SPACE])
            self.assertEqual(rc, 0)
            evs = st.read_events(state)
            self.assertEqual(len(evs), 1)
            self.assertEqual(evs[0]["trial_type"], "merge")
            self.assertEqual(evs[0]["notes"], "test merge")


if __name__ == "__main__":
    unittest.main()
