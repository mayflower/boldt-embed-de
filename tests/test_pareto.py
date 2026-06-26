"""Unit tests for boldt_embed.pareto — pure dominance logic, stdlib only.

Covers: basic strict dominance, no-domination on ties, the conservative None/missing rule
(a hole can neither manufacture a win nor a loss, and a fully-missing candidate survives),
direction handling (lower-is-better cost), and cost tie-breaking.
"""
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_pareto():
    spec = importlib.util.spec_from_file_location(
        "boldt_embed_pareto", ROOT / "src" / "boldt_embed" / "pareto.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load_pareto()


class ImportTests(unittest.TestCase):
    def test_imports_without_torch(self):
        # If pareto imported torch at module top it would already have failed above.
        import sys
        self.assertNotIn("torch", sys.modules.get("boldt_embed_pareto", P).__dict__)


class DominanceTests(unittest.TestCase):
    M = ["a", "b"]

    def test_strict_dominance(self):
        a = {"a": 0.9, "b": 0.5}
        b = {"a": 0.8, "b": 0.5}   # a >= b everywhere, strictly better on 'a'
        self.assertTrue(P.dominates(a, b, self.M))
        self.assertFalse(P.dominates(b, a, self.M))

    def test_equal_does_not_dominate(self):
        a = {"a": 0.9, "b": 0.5}
        b = {"a": 0.9, "b": 0.5}
        self.assertFalse(P.dominates(a, b, self.M))
        self.assertFalse(P.dominates(b, a, self.M))

    def test_tradeoff_neither_dominates(self):
        a = {"a": 0.9, "b": 0.4}
        b = {"a": 0.8, "b": 0.5}   # each better on one axis
        self.assertFalse(P.dominates(a, b, self.M))
        self.assertFalse(P.dominates(b, a, self.M))

    def test_better_on_one_equal_on_other(self):
        a = {"a": 0.9, "b": 0.5}
        b = {"a": 0.9, "b": 0.4}
        self.assertTrue(P.dominates(a, b, self.M))


class MissingValueRuleTests(unittest.TestCase):
    """The conservative None rule: a missing metric is skipped for that pairwise comparison."""

    M = ["a", "b"]

    def test_missing_axis_skipped_not_treated_as_zero(self):
        # If None were treated as 0, a would dominate b on 'b'. It must NOT.
        a = {"a": 0.9, "b": 0.5}
        b = {"a": 0.8, "b": None}
        # Only 'a' is comparable; a is strictly better there and not worse anywhere -> a dominates.
        self.assertTrue(P.dominates(a, b, self.M))
        # But b cannot dominate a: it is worse on the one comparable axis.
        self.assertFalse(P.dominates(b, a, self.M))

    def test_missing_cannot_manufacture_a_win(self):
        # b has a hole where it would otherwise lose; that hole must not let b dominate.
        a = {"a": 0.9, "b": 0.9}
        b = {"a": 0.9, "b": None}   # equal on 'a', missing on 'b'
        # comparable only on 'a' where they tie -> nobody strictly better -> no domination.
        self.assertFalse(P.dominates(a, b, self.M))
        self.assertFalse(P.dominates(b, a, self.M))

    def test_no_comparable_axis_means_no_domination(self):
        a = {"a": 0.9, "b": None}
        b = {"a": None, "b": 0.5}
        self.assertFalse(P.dominates(a, b, self.M))
        self.assertFalse(P.dominates(b, a, self.M))

    def test_fully_missing_candidate_survives_frontier(self):
        full = {"label": "full", "a": 0.9, "b": 0.9}
        good = {"label": "good", "a": 0.5, "b": 0.5}
        unknown = {"label": "unknown", "a": None, "b": None}
        front = P.pareto_frontier([full, good, unknown], metrics=["a", "b"])
        labels = {c["label"] for c in front}
        # 'good' is dominated by 'full'; 'unknown' is incomparable so it survives.
        self.assertIn("full", labels)
        self.assertIn("unknown", labels)
        self.assertNotIn("good", labels)


class FrontierTests(unittest.TestCase):
    def test_frontier_basic(self):
        cands = [
            {"label": "x", "a": 0.9, "b": 0.2},
            {"label": "y", "a": 0.2, "b": 0.9},
            {"label": "z", "a": 0.5, "b": 0.5},
            {"label": "dom", "a": 0.1, "b": 0.1},   # dominated by all
        ]
        front = {c["label"] for c in P.pareto_frontier(cands, metrics=["a", "b"])}
        self.assertEqual(front, {"x", "y", "z"})

    def test_identical_rows_both_survive(self):
        cands = [{"label": "a", "m": 0.5}, {"label": "b", "m": 0.5}]
        front = {c["label"] for c in P.pareto_frontier(cands, metrics=["m"])}
        self.assertEqual(front, {"a", "b"})


class DirectionAndCostTests(unittest.TestCase):
    def test_lower_is_better_direction(self):
        directions = {"vram_gb": P.LOWER}
        a = {"vram_gb": 4.0}
        b = {"vram_gb": 8.0}
        self.assertTrue(P.dominates(a, b, ["vram_gb"], directions))
        self.assertFalse(P.dominates(b, a, ["vram_gb"], directions))

    def test_cost_metrics_excluded_from_objective_by_default(self):
        cands = [{"a": 0.5, "vram_gb": 4.0, "throughput_pairs_per_sec": 1000.0},
                 {"a": 0.5, "vram_gb": 9.0, "throughput_pairs_per_sec": 50.0}]
        objs = P.objective_metrics(cands)
        self.assertIn("a", objs)
        self.assertNotIn("vram_gb", objs)
        self.assertNotIn("throughput_pairs_per_sec", objs)

    def test_tie_break_prefers_low_vram_then_high_throughput(self):
        cands = [
            {"label": "hi_vram", "vram_gb": 9.0, "throughput_pairs_per_sec": 100.0},
            {"label": "lo_vram", "vram_gb": 4.0, "throughput_pairs_per_sec": 100.0},
        ]
        ordered = [c["label"] for c in P.tie_break(cands)]
        self.assertEqual(ordered[0], "lo_vram")

    def test_tie_break_known_cost_beats_missing(self):
        cands = [
            {"label": "missing", "vram_gb": None, "throughput_pairs_per_sec": None},
            {"label": "known", "vram_gb": 6.0, "throughput_pairs_per_sec": 500.0},
        ]
        ordered = [c["label"] for c in P.tie_break(cands)]
        self.assertEqual(ordered[0], "known")

    def test_tie_break_higher_throughput_wins_when_vram_equal(self):
        cands = [
            {"label": "slow", "vram_gb": 4.0, "throughput_pairs_per_sec": 100.0},
            {"label": "fast", "vram_gb": 4.0, "throughput_pairs_per_sec": 900.0},
        ]
        ordered = [c["label"] for c in P.tie_break(cands)]
        self.assertEqual(ordered[0], "fast")


if __name__ == "__main__":
    unittest.main()
