#!/usr/bin/env python3
"""Build an EmbedFilter projection basis from a model's unembedding matrix via SVD (lazy ML).

``torch``/``transformers`` are imported INSIDE ``main`` only. ``--dry-run`` plans the spec from
``--hidden-dim`` + ``--tau`` and imports no ML. A real build saves ``<out>/basis.pt``
([hidden_dim, keep_dim], float32) + ``<out>/metadata.json`` (full provenance).

    python scripts/build_embed_filter.py --model Boldt/Boldt-DC-350M --tau 2 \
        --out outputs/embedfilter/boldt-dc-350m_tau2
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import embed_filter as EF  # noqa: E402
from boldt_embed import experiment_registry as registry  # noqa: E402  (stdlib git/version helpers)


def plan_build(hidden_dim: int, tau: int, model: str, out: Optional[str]) -> Dict[str, Any]:
    """Stdlib dry-run planner: resolve the spec without loading any model."""
    spec = EF.select_bulk_slice(hidden_dim, tau)
    return {
        "model": model,
        "out": out or f"outputs/embedfilter/dry_tau{tau}",
        "spec": {"hidden_dim": spec.hidden_dim, "tau": spec.tau, "keep_dim": spec.keep_dim,
                 "left": spec.left, "right": spec.right, "strategy": spec.strategy},
        "basis_shape": [spec.hidden_dim, spec.keep_dim],
    }


def _command(argv: Optional[List[str]]) -> str:
    return "python " + " ".join([sys.argv[0]] + (argv if argv is not None else sys.argv[1:]))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Boldt/Boldt-DC-350M")
    ap.add_argument("--tau", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--svd-dtype", default="float32")
    ap.add_argument("--hidden-dim", type=int, default=None, help="for --dry-run planning only")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.dry_run:
        if args.hidden_dim is None:
            print("error: --dry-run requires --hidden-dim", file=sys.stderr)
            return 2
        plan = plan_build(args.hidden_dim, args.tau, args.model, args.out)
        print(json.dumps({"status": "dry_run", **plan}, ensure_ascii=False, indent=2))
        return 0

    if not args.out:
        print("error: --out is required for a real build", file=sys.stderr)
        return 2

    # ---------------------------------------------------------------- real build (lazy ML)
    import torch
    from transformers import AutoModelForCausalLM

    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    dev = "cuda" if use_cuda else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    warnings: List[str] = []
    oe = model.get_output_embeddings()
    lm = getattr(model, "lm_head", None)
    if oe is not None and getattr(oe, "weight", None) is not None:
        W, source = oe.weight, "output_embeddings"
    elif getattr(lm, "weight", None) is not None:
        W, source = lm.weight, "lm_head"
    else:
        W, source = model.get_input_embeddings().weight, "tied_input_embeddings"
        warnings.append("no output embeddings / lm_head found — used tied INPUT embeddings")

    vocab, hidden = int(W.shape[0]), int(W.shape[1])
    spec = EF.select_bulk_slice(hidden, args.tau)

    svd_dtype = getattr(torch, args.svd_dtype)
    Wm = W.detach().to(dev, dtype=svd_dtype)
    with torch.no_grad():
        _U, S, Vh = torch.linalg.svd(Wm, full_matrices=False)   # Vh: [hidden, hidden]
        basis = Vh[spec.left:spec.right].T.contiguous().to("cpu", dtype=torch.float32)  # [H, K]

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(basis, out / "basis.pt")

    s = S.detach().to("cpu").float().tolist()
    total_energy = sum(v * v for v in s) or 1.0
    sigma_stats = {
        "n": len(s), "max": s[0], "min": s[-1],
        "kept_first": s[spec.left], "kept_last": s[spec.right - 1],
        "energy_kept_frac": round(sum(v * v for v in s[spec.left:spec.right]) / total_energy, 6),
    }
    meta = EF.metadata_for_spec(
        spec, model=args.model, source_matrix=source, vocab_size=vocab,
        extra={
            "device": dev, "svd_dtype": args.svd_dtype, "basis_shape": list(basis.shape),
            "singular_value_stats": sigma_stats,
            "torch": registry._pkg_version("torch"),
            "transformers": registry._pkg_version("transformers"),
            "command": _command(argv), "commit": registry.current_git_commit(),
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "warnings": warnings,
        })
    problems = EF.validate_embed_filter_metadata(meta)
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")

    print(f"[embedfilter] {args.model} tau={args.tau} -> basis {list(basis.shape)} "
          f"source={source} energy_kept={sigma_stats['energy_kept_frac']:.4f} -> {out}/basis.pt")
    if warnings:
        print("  warnings: " + "; ".join(warnings), file=sys.stderr)
    if problems:
        print("  metadata problems: " + "; ".join(problems), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
