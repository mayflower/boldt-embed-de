"""Unit tests for the pure-stdlib merge math (Prompt 08).

Tiny artificial state dicts only — no torch, no weights, no IO.
"""
import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import merge_methods  # noqa: E402


class TestMergeMethods(unittest.TestCase):
    def test_no_torch_imported(self):
        # Checked in a FRESH subprocess so unittest-discover's torch-using modules can't pollute
        # the shared sys.modules (a bare `'torch' not in sys.modules` is order-dependent).
        sys.path.insert(0, str(ROOT / "tests"))
        from torch_free import module_is_torch_free
        self.assertTrue(module_is_torch_free("boldt_embed.merge_methods"))

    def test_mean(self):
        p1 = {"w": [1.0, 2.0, 3.0]}
        p2 = {"w": [3.0, 4.0, 5.0]}
        out = merge_methods.mean([p1, p2])
        self.assertEqual(out["w"], [2.0, 3.0, 4.0])

    def test_weighted_mean_applies_and_normalizes(self):
        p1 = {"w": [0.0, 0.0]}
        p2 = {"w": [4.0, 8.0]}
        # weights [1, 3] normalize to [0.25, 0.75] -> 0.75 * p2
        out = merge_methods.weighted_mean([p1, p2], [1.0, 3.0])
        self.assertAlmostEqual(out["w"][0], 3.0)
        self.assertAlmostEqual(out["w"][1], 6.0)
        # unnormalized weights summing to >1 give the same answer as normalized ones
        out2 = merge_methods.weighted_mean([p1, p2], [2.0, 6.0])
        self.assertAlmostEqual(out["w"][0], out2["w"][0])

    def test_weighted_mean_zero_sum_raises(self):
        with self.assertRaises(ValueError):
            merge_methods.weighted_mean([{"w": [1.0]}, {"w": [2.0]}], [0.0, 0.0])

    def test_slerp_endpoints_and_midpoint(self):
        a = {"w": [1.0, 0.0]}
        b = {"w": [0.0, 1.0]}
        at0 = merge_methods.slerp_pairwise([a, b], 0.0)
        at1 = merge_methods.slerp_pairwise([a, b], 1.0)
        mid = merge_methods.slerp_pairwise([a, b], 0.5)
        self.assertAlmostEqual(at0["w"][0], 1.0)
        self.assertAlmostEqual(at0["w"][1], 0.0)
        self.assertAlmostEqual(at1["w"][0], 0.0)
        self.assertAlmostEqual(at1["w"][1], 1.0)
        # midpoint of orthogonal unit vectors is norm-preserving (slerp property)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in mid["w"])), 1.0)
        self.assertAlmostEqual(mid["w"][0], mid["w"][1])

    def test_slerp_colinear_lerp_fallback(self):
        a = {"w": [1.0, 0.0]}
        b = {"w": [2.0, 0.0]}  # colinear
        mid = merge_methods.slerp_pairwise([a, b], 0.5)
        self.assertAlmostEqual(mid["w"][0], 1.5)  # plain lerp midpoint

    def test_slerp_requires_two_parents(self):
        a = {"w": [1.0]}
        with self.assertRaises(ValueError):
            merge_methods.slerp_pairwise([a, a, a], 0.5)

    def test_task_vector_sum(self):
        base = {"w": [0.0, 0.0]}
        p1 = {"w": [1.0, 0.0]}  # delta [1, 0]
        p2 = {"w": [0.0, 2.0]}  # delta [0, 2]
        out = merge_methods.task_vector_sum([p1, p2], base)
        self.assertEqual(out["w"], [1.0, 2.0])
        # with a non-zero base the deltas are added back to it
        base2 = {"w": [10.0, 10.0]}
        p1b = {"w": [11.0, 10.0]}  # delta [1, 0]
        p2b = {"w": [10.0, 12.0]}  # delta [0, 2]
        out2 = merge_methods.task_vector_sum([p1b, p2b], base2)
        self.assertEqual(out2["w"], [11.0, 12.0])

    def test_ties_sign_election_and_density_trim(self):
        # base zero -> deltas == parent values. density 0.5 of 4 = keep top 2 by |magnitude| per parent.
        base = {"w": [0.0, 0.0, 0.0, 0.0]}
        p1 = {"w": [5.0, -1.0, 3.0, 0.2]}   # top2 by |.|: idx0(5), idx2(3) -> kept +5, +3
        p2 = {"w": [4.0, 2.0, -6.0, 0.1]}   # top2 by |.|: idx2(6), idx0(4) -> kept +4, -6
        out = merge_methods.ties([p1, p2], base, density=0.5)
        w = out["w"]
        # idx0: p1 +5, p2 +4 both kept & positive -> elected +, mean = 4.5
        self.assertAlmostEqual(w[0], 4.5)
        # idx1: neither kept (below density trim) -> 0
        self.assertAlmostEqual(w[1], 0.0)
        # idx2: p1 kept +3, p2 kept -6 -> sum -3 elects negative -> only -6 agrees -> -6
        self.assertAlmostEqual(w[2], -6.0)
        # idx3: neither kept (below density trim) -> 0
        self.assertAlmostEqual(w[3], 0.0)

    def test_ties_density_bounds(self):
        with self.assertRaises(ValueError):
            merge_methods.ties([{"w": [1.0]}, {"w": [1.0]}], {"w": [0.0]}, density=0.0)
        with self.assertRaises(ValueError):
            merge_methods.ties([{"w": [1.0]}, {"w": [1.0]}], {"w": [0.0]}, density=1.5)

    def test_dare_deterministic_and_rescale(self):
        base = {"w": [0.0] * 8}
        p1 = {"w": [1.0] * 8}
        # same seed -> identical output (deterministic mask)
        a = merge_methods.dare_linear([p1], base, density=0.5, rescale=True, seed=42)
        b = merge_methods.dare_linear([p1], base, density=0.5, rescale=True, seed=42)
        self.assertEqual(a["w"], b["w"])
        # surviving entries are rescaled to delta/density = 1/0.5 = 2.0; dropped are 0.
        for v in a["w"]:
            self.assertIn(round(v, 6), (0.0, 2.0))
        # at least one survives, at least one dropped at density 0.5 over 8 entries (statistically)
        self.assertTrue(any(v != 0.0 for v in a["w"]))
        # different seed -> (very likely) different mask
        c = merge_methods.dare_linear([p1], base, density=0.5, rescale=True, seed=7)
        self.assertNotEqual(a["w"], c["w"])
        # no rescale -> survivors keep delta magnitude 1.0
        d = merge_methods.dare_linear([p1], base, density=0.5, rescale=False, seed=42)
        for v in d["w"]:
            self.assertIn(round(v, 6), (0.0, 1.0))

    def test_layerwise_weighted_mean(self):
        p1 = {"a": [0.0], "b": [0.0]}
        p2 = {"a": [10.0], "b": [10.0]}
        out = merge_methods.layerwise_weighted_mean(
            [p1, p2], {"a": [0.0, 1.0]}  # 'a' fully from p2; 'b' falls back to uniform
        )
        self.assertAlmostEqual(out["a"][0], 10.0)
        self.assertAlmostEqual(out["b"][0], 5.0)

    def test_key_mismatch_raises(self):
        with self.assertRaises(ValueError):
            merge_methods.mean([{"w": [1.0]}, {"x": [1.0]}])

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            merge_methods.mean([{"w": [1.0, 2.0]}, {"w": [1.0]}])

    def test_base_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            merge_methods.task_vector_sum([{"w": [1.0, 2.0]}, {"w": [1.0, 2.0]}], {"w": [0.0]})

    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError):
            merge_methods.weighted_mean([{"w": [1.0]}, {"w": [2.0]}], [-1.0, 2.0])

    def test_dare_mask_reproducible_across_processes(self):
        # The DARE keep-mask must be identical regardless of PYTHONHASHSEED — otherwise a 'seeded'
        # DARE merge silently differs per process. Run the same mask in two subprocesses with
        # DIFFERENT hash seeds and assert byte-identical output.
        import subprocess as _sp
        code = ("import sys; sys.path.insert(0, %r);"
                "from boldt_embed.merge_methods import _seeded_mask;"
                "print(''.join('1' if b else '0' for b in _seeded_mask(42, 'encoder.w', 64, 0.5)))"
                % str(ROOT / "src"))
        import os as _os
        outs = []
        for hs in ("0", "1"):
            env = dict(_os.environ, PYTHONHASHSEED=hs)
            r = _sp.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
            outs.append(r.stdout.strip())
        self.assertEqual(outs[0], outs[1], "DARE mask differs across PYTHONHASHSEED — not reproducible")


if __name__ == "__main__":
    unittest.main()
