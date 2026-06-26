"""Helper: assert a module/script is stdlib-pure (does not pull torch/transformers/numpy AT IMPORT).

A bare ``assert 'torch' not in sys.modules`` is unreliable under ``unittest discover``: the collection
phase imports torch-using test modules into the SAME process, polluting sys.modules for everyone. The
honest, order-independent check is to import the target in a FRESH subprocess and inspect that
process's sys.modules. These helpers do exactly that.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_HEAVY = ("torch", "transformers", "numpy")


def is_torch_free(snippet: str) -> bool:
    """Run ``snippet`` in a fresh interpreter (src on path); True iff no heavy ML dep got imported."""
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        + snippet + "\n"
        "import sys as _s\n"
        f"_bad = [m for m in {_HEAVY!r} if m in _s.modules]\n"
        "raise SystemExit(13 if _bad else 0)\n"
    )
    return subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                          capture_output=True).returncode == 0


def module_is_torch_free(dotted: str) -> bool:
    return is_torch_free(f"import {dotted}")


def script_is_torch_free(rel_path: str) -> bool:
    return is_torch_free(
        "import importlib.util as u\n"
        f"s = u.spec_from_file_location('m', {rel_path!r})\n"
        "m = u.module_from_spec(s); s.loader.exec_module(m)")
