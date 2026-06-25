#!/usr/bin/env python3
"""Merge N SentenceTransformers checkpoints into one — for the v8 specialist→merge lever.

Generalizes `scripts/slerp_merge.py` (2-model SLERP) to:
  - `--method mean`   : (weighted) uniform model-soup over N models (Wortsman 2022 — robust for N>2)
  - `--method slerp`  : spherical interpolation of exactly 2 models (delegates to the SLERP math)

The point of merging is to escape the composition trade-off: a wiki/MIRACL specialist + a
legal/GerDaLIR specialist + a FAQ specialist, merged, can inherit each one's strong task — IF the
checkpoints share a basin (train them from a common warm-start). All inputs must have IDENTICAL
parameter shapes. Non-weight files (config/tokenizer/pooling) are copied from the first model so the
output loads as an ST model. Needs torch + safetensors; NOT part of the stdlib gates.

This script only PRODUCES a merged checkpoint — evaluating/gating it (frontier gate) is done by the
/ar-merge command, never silently here.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _slerp(t: float, w0, w1, eps: float = 1e-8):
    import torch

    shape, dtype = w0.shape, w0.dtype
    a, b = w0.flatten().float(), w1.flatten().float()
    na, nb = a / (a.norm() + eps), b / (b.norm() + eps)
    dot = torch.dot(na, nb).clamp(-1 + 1e-7, 1 - 1e-7)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    if so.abs() < 1e-6:                       # nearly colinear → LERP
        res = (1 - t) * a + t * b
    else:
        res = (torch.sin((1 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b
    return res.reshape(shape).to(dtype)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", required=True, help="comma-separated ST checkpoint dirs (>=2)")
    ap.add_argument("--weights", default=None, help="comma-separated weights (default: uniform)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["mean", "slerp"], default="mean")
    ap.add_argument("--t", type=float, default=0.5, help="slerp interp factor (2-model only)")
    ap.add_argument("--weights-file", default="model.safetensors")
    args = ap.parse_args()

    import torch
    from safetensors.torch import load_file, save_file

    dirs = [Path(p.strip()) for p in args.models.split(",") if p.strip()]
    if len(dirs) < 2:
        raise SystemExit("need >=2 models to merge")
    if args.method == "slerp" and len(dirs) != 2:
        raise SystemExit("--method slerp requires exactly 2 models (use mean for N>2)")
    if args.weights:
        w = [float(x) for x in args.weights.split(",")]
        if len(w) != len(dirs):
            raise SystemExit(f"--weights has {len(w)} entries for {len(dirs)} models")
    else:
        w = [1.0 / len(dirs)] * len(dirs)
    s = sum(w) or 1.0
    w = [x / s for x in w]

    sds = [load_file(str(d / args.weights_file)) for d in dirs]
    keys = set(sds[0])
    for i, sd in enumerate(sds[1:], 1):
        if set(sd) != keys:
            raise SystemExit(f"key mismatch between {dirs[0]} and {dirs[i]}")

    merged = {}
    for k in sds[0]:
        if args.method == "slerp":
            merged[k] = _slerp(args.t, sds[0][k], sds[1][k])
        else:
            acc = None
            for wi, sd in zip(w, sds):
                if sd[k].shape != sds[0][k].shape:
                    raise SystemExit(f"shape mismatch on {k}")
                term = sd[k].float() * wi
                acc = term if acc is None else acc + term
            merged[k] = acc.to(sds[0][k].dtype)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(out / args.weights_file), metadata={"format": "pt"})
    for p in dirs[0].iterdir():                 # copy ST scaffolding from the first model
        if p.name == args.weights_file or p.name.startswith("checkpoint-"):
            continue
        (shutil.copytree if p.is_dir() else shutil.copy2)(p, out / p.name)

    desc = f"slerp(t={args.t})" if args.method == "slerp" else f"mean(weights={[round(x,3) for x in w]})"
    print(f"merged {len(merged)} tensors via {desc} -> {out}")
    for d, wi in zip(dirs, w):
        print(f"  {wi:.3f}  {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
