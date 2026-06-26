#!/usr/bin/env python3
"""Merge-search orchestrator for the v8 specialist→merge lever (Prompt 08).

Splits cleanly into two halves:
  * the MATH lives in ``src/boldt_embed/merge_methods.py`` (pure stdlib, state-dict-of-lists);
  * the IO lives HERE — loading checkpoints (safetensors / pytorch_model.bin), flattening each
    tensor to a list, calling the math, reshaping back, writing the merged checkpoint and copying
    the non-weight SentenceTransformers scaffolding from a reference parent.

Fail-closed contract:
  * ``--dry-run`` (the default) imports NO torch. It only expands and lists the planned
    (method, params) candidates as JSON — no checkpoint is touched.
  * Real merges require ``--real --allow-merge`` together.
  * A method that cannot be safely applied to the configured inputs is reported ``unsupported``
    WITH a reason (e.g. task-arithmetic methods without a ``warm_start``), never mis-merged.
  * Merged checkpoints go under ``--out`` (expected: ``outputs/merged/...``, gitignored). Weights
    are never committed. No benchmark claim is emitted without a saved eval output.

CLI:
  python scripts/ar_merge_search.py --config configs/autoresearch/merge_search_v8.json \
      --out outputs/merged/v8_merge_search --dry-run
  python scripts/ar_merge_search.py --config ... --out ... --real --allow-merge [--eval-fast]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import merge_methods  # noqa: E402  (pure stdlib — safe to import at top level)

# Methods that operate relative to the common warm-start basin; unsupported without one.
_BASE_RELATIVE = {"task_vector_sum", "ties", "dare_linear"}


# --------------------------------------------------------------------------------------------
# config + candidate planning  (pure stdlib — used by dry-run AND real)
# --------------------------------------------------------------------------------------------
def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def plan_candidates(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Expand ``config["methods"]`` grids into concrete (method, params) candidates.

    Returns ``(supported, unsupported)``. Each supported candidate is a dict with ``method`` and a
    ``params`` block; each unsupported one carries ``method`` + ``reason``.
    """
    parents = config.get("parents", [])
    warm_start = config.get("warm_start")
    n = len(parents)
    supported: List[Dict[str, Any]] = []
    unsupported: List[Dict[str, Any]] = []

    for spec in config.get("methods", []):
        name = spec.get("name")
        if name not in merge_methods.METHODS:
            unsupported.append({"method": name, "reason": f"unknown method {name!r}"})
            continue
        if name in _BASE_RELATIVE and not warm_start:
            unsupported.append(
                {"method": name, "reason": "requires a 'warm_start' base in the config (task "
                                           "arithmetic is undefined without a common basin)"}
            )
            continue

        if name == "mean":
            supported.append({"method": name, "params": {}})

        elif name == "weighted_mean":
            grid = spec.get("weights_grid", [[1.0 / n] * n]) if n else []
            for weights in grid:
                if len(weights) != n:
                    unsupported.append(
                        {"method": name, "params": {"weights": weights},
                         "reason": f"weights length {len(weights)} != {n} parents"}
                    )
                    continue
                supported.append({"method": name, "params": {"weights": list(weights)}})

        elif name == "slerp_pairwise":
            if n != 2:
                unsupported.append(
                    {"method": name, "reason": f"requires exactly 2 parents, config has {n}"}
                )
                continue
            for t in spec.get("t_grid", [0.5]):
                supported.append({"method": name, "params": {"t": float(t)}})

        elif name == "task_vector_sum":
            supported.append({"method": name, "params": {"base": "warm_start"}})

        elif name == "ties":
            for d in spec.get("density_grid", [0.5]):
                supported.append(
                    {"method": name, "params": {"density": float(d), "base": "warm_start"}}
                )

        elif name == "dare_linear":
            rescale = bool(spec.get("rescale", True))
            seed = int(spec.get("seed", 0))
            for d in spec.get("density_grid", [0.5]):
                supported.append(
                    {"method": name,
                     "params": {"density": float(d), "rescale": rescale, "seed": seed,
                                "base": "warm_start"}}
                )

        elif name == "layerwise_weighted_mean":
            # Per-key weights are too large/structural for a grid; require an explicit mapping.
            wpk = spec.get("weights_per_key")
            if not wpk:
                unsupported.append(
                    {"method": name,
                     "reason": "requires an explicit 'weights_per_key' mapping in the config"}
                )
                continue
            supported.append({"method": name, "params": {"weights_per_key": wpk}})

    return supported, unsupported


def _candidate_label(cand: Dict[str, Any], idx: int) -> str:
    name = cand["method"]
    p = cand.get("params", {})
    bits: List[str] = []
    if "weights" in p:
        bits.append("w" + "-".join(str(round(x, 3)) for x in p["weights"]))
    if "t" in p:
        bits.append(f"t{p['t']}")
    if "density" in p:
        bits.append(f"d{p['density']}")
    if p.get("rescale") is False:
        bits.append("norescale")
    suffix = ("_" + "_".join(bits)) if bits else ""
    return f"{idx:02d}_{name}{suffix}"


def build_dry_run_report(config: Dict[str, Any], out: str | Path) -> Dict[str, Any]:
    supported, unsupported = plan_candidates(config)
    parents = config.get("parents", [])
    return {
        "name": config.get("name", "merge_search"),
        "out": str(out),
        "dry_run": True,
        "warm_start": config.get("warm_start"),
        "parents": [{"label": p.get("label"), "path": p.get("path")} for p in parents],
        "n_parents": len(parents),
        "planned_candidates": [
            {"label": _candidate_label(c, i), **c} for i, c in enumerate(supported)
        ],
        "unsupported": unsupported,
        "eval": config.get("eval", {}),
        "note": "dry-run: no checkpoints loaded, no torch imported. "
                "Run with --real --allow-merge to materialize merges.",
    }


# --------------------------------------------------------------------------------------------
# checkpoint IO  (torch imported LAZILY — never on the dry-run path)
# --------------------------------------------------------------------------------------------
def _weights_filename(ckpt_dir: Path) -> str:
    if (ckpt_dir / "model.safetensors").exists():
        return "model.safetensors"
    if (ckpt_dir / "pytorch_model.bin").exists():
        return "pytorch_model.bin"
    raise FileNotFoundError(
        f"no model.safetensors or pytorch_model.bin under {ckpt_dir}"
    )


def _load_state_dict(ckpt_dir: Path):
    """Load a checkpoint's tensors. Imports torch/safetensors lazily; returns a torch state dict."""
    fname = _weights_filename(ckpt_dir)
    path = ckpt_dir / fname
    if fname.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SystemExit(
                "safetensors required for real merges — install the [eval]/[train] extra"
            ) from exc
        return load_file(str(path)), fname
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("torch required for real merges — install the [train] extra") from exc
    return torch.load(str(path), map_location="cpu"), fname


def _flatten(state_dict) -> Dict[str, List[float]]:
    return {k: v.detach().float().flatten().tolist() for k, v in state_dict.items()}


def _reshape_into(reference_sd, flat: Dict[str, List[float]]):
    """Rebuild a torch state dict from flattened lists, using the reference tensors' shape+dtype."""
    import torch

    out = {}
    for k, ref in reference_sd.items():
        t = torch.tensor(flat[k], dtype=torch.float32).reshape(ref.shape)
        out[k] = t.to(ref.dtype)
    return out


def _save_state_dict(state_dict, out_dir: Path, fname: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if fname.endswith(".safetensors"):
        from safetensors.torch import save_file

        save_file(state_dict, str(out_dir / fname), metadata={"format": "pt"})
    else:
        import torch

        torch.save(state_dict, str(out_dir / fname))


def _copy_st_scaffolding(ref_dir: Path, out_dir: Path, weights_fname: str) -> List[str]:
    """Copy non-weight ST files (config/tokenizer/pooling) from a reference parent.

    Mirrors the pattern in scripts/model_soup.py: skip the weights file and training-checkpoint
    subdirs. Returns the list of copied entry names.
    """
    copied: List[str] = []
    for p in ref_dir.iterdir():
        if p.name in ("model.safetensors", "pytorch_model.bin") or p.name.startswith("checkpoint-"):
            continue
        if p.name == weights_fname:
            continue
        dest = out_dir / p.name
        (shutil.copytree if p.is_dir() else shutil.copy2)(p, dest)
        copied.append(p.name)
    return copied


def _apply_merge(cand: Dict[str, Any], flat_parents: List[Dict[str, List[float]]],
                 flat_base: Optional[Dict[str, List[float]]]) -> Dict[str, List[float]]:
    name = cand["method"]
    p = cand.get("params", {})
    if name == "mean":
        return merge_methods.mean(flat_parents)
    if name == "weighted_mean":
        return merge_methods.weighted_mean(flat_parents, p["weights"])
    if name == "slerp_pairwise":
        return merge_methods.slerp_pairwise(flat_parents, p["t"])
    if name == "task_vector_sum":
        return merge_methods.task_vector_sum(flat_parents, flat_base)
    if name == "ties":
        return merge_methods.ties(flat_parents, flat_base, p["density"])
    if name == "dare_linear":
        return merge_methods.dare_linear(
            flat_parents, flat_base, p["density"], rescale=p.get("rescale", True),
            seed=p.get("seed", 0),
        )
    if name == "layerwise_weighted_mean":
        return merge_methods.layerwise_weighted_mean(flat_parents, p["weights_per_key"])
    raise ValueError(f"no apply path for method {name!r}")


def run_real(config: Dict[str, Any], out_root: Path, eval_fast: bool) -> Dict[str, Any]:
    """Materialize every supported candidate. torch is imported lazily inside the loaders."""
    supported, unsupported = plan_candidates(config)
    parents = config.get("parents", [])
    warm_start = config.get("warm_start")
    if not parents:
        raise SystemExit("config has no parents to merge")

    ref_dir = Path(parents[0]["path"])  # reference parent for ST scaffolding + tensor shapes
    out_root.mkdir(parents=True, exist_ok=True)

    # load parents once (and the base if any base-relative method is planned)
    parent_sds, weights_fnames = [], []
    for par in parents:
        sd, fn = _load_state_dict(Path(par["path"]))
        parent_sds.append(sd)
        weights_fnames.append(fn)
    ref_sd = parent_sds[0]
    weights_fname = weights_fnames[0]
    flat_parents = [_flatten(sd) for sd in parent_sds]

    flat_base = None
    base_sd = None
    if warm_start and any(c["method"] in _BASE_RELATIVE for c in supported):
        base_sd, _ = _load_state_dict(Path(warm_start))
        flat_base = _flatten(base_sd)

    results: List[Dict[str, Any]] = []
    for i, cand in enumerate(supported):
        label = _candidate_label(cand, i)
        cand_dir = out_root / label
        merged_flat = _apply_merge(cand, flat_parents, flat_base)
        merged_sd = _reshape_into(ref_sd, merged_flat)
        _save_state_dict(merged_sd, cand_dir, weights_fname)
        copied = _copy_st_scaffolding(ref_dir, cand_dir, weights_fname)

        merge_manifest = {
            "label": label,
            "method": cand["method"],
            "params": cand.get("params", {}),
            "weights_file": weights_fname,
            "n_tensors": len(merged_sd),
            "copied_st_files": copied,
            "eval": {"status": "skipped", "note": "no eval command wired in this PR"},
        }
        if eval_fast:
            # Stub per the PR spec: a fast proxy eval is not wired here; record intent, no claim.
            merge_manifest["eval"] = {
                "status": "planned",
                "note": "--eval-fast requested but no fast-proxy eval command is wired in this PR; "
                        "no benchmark claim is made without a saved eval output.",
            }
        with open(cand_dir / "merge_manifest.json", "w", encoding="utf-8") as fh:
            json.dump(merge_manifest, fh, indent=2)
        parent_manifest = {
            "warm_start": warm_start,
            "reference_parent": parents[0].get("label"),
            "parents": [
                {"label": par.get("label"), "path": par.get("path"),
                 "weights_file": weights_fnames[j]}
                for j, par in enumerate(parents)
            ],
        }
        with open(cand_dir / "parent_manifest.json", "w", encoding="utf-8") as fh:
            json.dump(parent_manifest, fh, indent=2)
        results.append({"label": label, "method": cand["method"], "out": str(cand_dir),
                        "eval": merge_manifest["eval"]})

    report = {
        "name": config.get("name", "merge_search"),
        "out": str(out_root),
        "dry_run": False,
        "merged": results,
        "unsupported": unsupported,
        "eval_fast": eval_fast,
    }
    with open(out_root / "merge_search_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="merge-search config JSON")
    ap.add_argument("--out", required=True, help="output root (expected under outputs/merged/)")
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="(default) list planned candidates as JSON; loads nothing, imports no torch")
    ap.add_argument("--real", action="store_true", help="actually run merges (needs --allow-merge)")
    ap.add_argument("--allow-merge", action="store_true",
                    help="safety interlock required alongside --real")
    ap.add_argument("--eval-fast", action="store_true",
                    help="request a fast-proxy eval after each merge (stubbed in this PR)")
    args = ap.parse_args(argv)

    config = load_config(args.config)

    if args.real:
        if not args.allow_merge:
            print(json.dumps(
                {"error": "refusing real merge without --allow-merge (fail-closed interlock)"},
                indent=2,
            ))
            return 2
        report = run_real(config, Path(args.out), eval_fast=args.eval_fast)
        print(json.dumps(report, indent=2))
        return 0

    # default / explicit dry-run: NO torch import on this path.
    report = build_dry_run_report(config, args.out)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
