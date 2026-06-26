"""P03 — AutoResearch state machine (stdlib): event log + deterministic next-trial ladder."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import autoresearch_state as st  # noqa: E402

FIXED_GIT = {"commit": "deadbeef", "dirty": False}


def _ev(trial_type, status="ok", i=0):
    return st.new_event(trial_type, status, event_id=f"e{i}",
                        timestamp_utc="2026-06-25T00:00:00+00:00", git=FIXED_GIT)


class TestEventIO(unittest.TestCase):
    def test_no_torch(self):
        import pathlib as _p
        sys.path.insert(0, str(_p.Path(__file__).resolve().parent))
        from torch_free import module_is_torch_free
        self.assertTrue(module_is_torch_free("boldt_embed.autoresearch_state"))

    def test_append_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "events.jsonl"
            st.append_event(_ev("data_mix", i=1), p)
            st.append_event(_ev("dense", i=2), p)
            evs = st.read_events(p)
            self.assertEqual([e["trial_type"] for e in evs], ["data_mix", "dense"])
            self.assertEqual(evs[0]["event_id"], "e1")

    def test_read_missing_is_empty(self):
        self.assertEqual(st.read_events(Path("/nonexistent/events.jsonl")), [])

    def test_event_schema_complete(self):
        e = _ev("merge")
        for k in ("event_id", "timestamp_utc", "trial_type", "status", "parent_artifacts",
                  "input_artifacts", "output_artifacts", "config", "metrics", "gates", "notes", "git"):
            self.assertIn(k, e)
        self.assertEqual(e["git"], FIXED_GIT)

    def test_invalid_trial_type_raises(self):
        with self.assertRaises(ValueError):
            st.new_event("nope")

    def test_invalid_status_raises(self):
        with self.assertRaises(ValueError):
            st.new_event("dense", "weird")


class TestDecisionLadder(unittest.TestCase):
    def test_empty_plans_data_mix(self):
        self.assertEqual(st.decide_next([]), "data_mix")

    def test_full_ladder_progression(self):
        evs = []
        steps = [
            ("data_mix", "dense"),
            ("dense", "hardneg_refresh"),
            ("hardneg_refresh", "specialist"),
            ("specialist", "specialist"),   # only 1 specialist -> still specialist
            ("specialist", "merge"),        # 2 specialists -> merge
            ("merge", "distill"),
            ("distill", "mteb"),
            ("mteb", "promotion"),
        ]
        for i, (did, expected_next) in enumerate(steps):
            evs.append(_ev(did, "ok", i))
            self.assertEqual(st.decide_next(evs), expected_next,
                             f"after {did} (#{i}) expected {expected_next}")
        evs.append(_ev("promotion", "ok", 99))
        self.assertIsNone(st.decide_next(evs))

    def test_failed_events_do_not_advance(self):
        # a failed data_mix does not satisfy the first ladder step
        self.assertEqual(st.decide_next([_ev("data_mix", "fail")]), "data_mix")

    def test_success_counts_only_ok(self):
        evs = [_ev("dense", "ok"), _ev("dense", "fail"), _ev("dense", "planned")]
        self.assertEqual(st.success_counts(evs)["dense"], 1)

    def test_summary_shape(self):
        s = st.summarize([_ev("data_mix", "ok")])
        self.assertEqual(s["n_events"], 1)
        self.assertEqual(s["next_trial_type"], "dense")
        self.assertIn("by_type_status", s)


if __name__ == "__main__":
    unittest.main()
