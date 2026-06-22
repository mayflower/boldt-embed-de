"""Tests for the EmbedFilter basis builder (dry-run planner + import safety; no real SVD in CI)."""
import contextlib
import importlib.util
import io
import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load("build_embed_filter")


class PlannerTests(unittest.TestCase):
    def test_plan_build_shapes(self):
        plan = B.plan_build(1024, 4, "Boldt/Boldt-DC-350M", None)
        self.assertEqual(plan["spec"]["keep_dim"], 256)
        self.assertEqual(plan["spec"]["left"], 384)
        self.assertEqual(plan["spec"]["right"], 640)
        self.assertEqual(plan["basis_shape"], [1024, 256])

    def test_dry_run_main_ok(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = B.main(["--dry-run", "--hidden-dim", "1024", "--tau", "2"])
        self.assertEqual(rc, 0)
        out = json.loads(buf.getvalue())
        self.assertEqual(out["status"], "dry_run")
        self.assertEqual(out["basis_shape"], [1024, 512])

    def test_dry_run_requires_hidden_dim(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(B.main(["--dry-run", "--tau", "2"]), 2)

    def test_real_requires_out(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(B.main(["--tau", "2"]), 2)


class ImportSafetyTests(unittest.TestCase):
    def test_dry_run_imports_no_ml(self):
        # fresh interpreter: a dry-run build must not import torch/transformers
        code = (
            "import sys, importlib.util, pathlib;"
            "root=pathlib.Path('.').resolve();"
            "spec=importlib.util.spec_from_file_location('b','scripts/build_embed_filter.py');"
            "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
            "rc=m.main(['--dry-run','--hidden-dim','1024','--tau','8']);"
            "assert rc==0, rc;"
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules, 'ML imported'"
        )
        r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
