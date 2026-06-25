#!/usr/bin/env python3
"""Spherical-linear-interpolation (SLERP) merge of two SentenceTransformers checkpoints.

The Qwen3-Embedding report attributes a robustness/generalization gain to SLERP-merging multiple
fine-tuning checkpoints (it lifted their 0.6B model ~+1.77 MMTEB). This merges two checkpoints with
IDENTICAL parameter shapes tensor-by-tensor: each tensor is treated as a vector on a hypersphere and
interpolated by angle (falls back to LERP when the two are nearly colinear). Non-weight files
(config, tokenizer, pooling) are copied from --model-a so the output is a loadable ST model.

Needs torch + safetensors (the [eval]/[train] extra); not part of the stdlib gates.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def slerp(t: float, w0, w1, eps: float = 1e-8):
    import torch

    shape, dtype = w0.shape, w0.dtype
    a, b = w0.flatten().float(), w1.flatten().float()
    na, nb = a / (a.norm() + eps), b / (b.norm() + eps)
    dot = torch.dot(na, nb).clamp(-1 + 1e-7, 1 - 1e-7)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    if so.abs() < 1e-6:                       # nearly colinear → plain LERP
        res = (1 - t) * a + t * b
    else:
        res = (torch.sin((1 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b
    return res.reshape(shape).to(dtype)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-a", required=True, help="ST checkpoint dir (config copied from here)")
    ap.add_argument("--model-b", required=True, help="ST checkpoint dir (same param shapes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t", type=float, default=0.5, help="interp factor: 0=A, 1=B")
    ap.add_argument("--weights", default="model.safetensors")
    args = ap.parse_args()

    from safetensors.torch import load_file, save_file

    a_dir, b_dir, out = Path(args.model_a), Path(args.model_b), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sd_a = load_file(str(a_dir / args.weights))
    sd_b = load_file(str(b_dir / args.weights))
    if set(sd_a) != set(sd_b):
        raise SystemExit(f"key mismatch: {len(set(sd_a) ^ set(sd_b))} differing keys")

    merged, lerp_count = {}, 0
    for k, wa in sd_a.items():
        wb = sd_b[k]
        if wa.shape != wb.shape:
            raise SystemExit(f"shape mismatch on {k}: {wa.shape} vs {wb.shape}")
        merged[k] = slerp(args.t, wa, wb)
    save_file(merged, str(out / args.weights), metadata={"format": "pt"})

    # copy the ST scaffolding (config/tokenizer/pooling) from A; skip weights + training checkpoints
    for p in a_dir.iterdir():
        if p.name == args.weights or p.name.startswith("checkpoint-"):
            continue
        (shutil.copytree if p.is_dir() else shutil.copy2)(p, out / p.name)

    print(f"SLERP(t={args.t}) merged {len(merged)} tensors -> {out}")
    print(f"  A={a_dir}\n  B={b_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
