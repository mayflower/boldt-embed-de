#!/usr/bin/env python3
"""Apply a bounded rerank policy to already-scored candidate lists (stdlib, no ML). Default mode is
``policy_gated`` (safe, bounded). Raw always-rerank is DISABLED unless ``--allow-raw-rerank-dangerous``
is passed. Policy decisions use only observable per-candidate fields — never qrels/labels.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import policy_reranker as PR  # noqa: E402
from boldt_embed.policy_config import load_policy  # noqa: E402


def _read(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=list(PR.MODES), default="policy_gated")
    ap.add_argument("--allow-raw-rerank-dangerous", action="store_true",
                    help="REQUIRED to use --mode raw_rerank; raw always-rerank is unsafe")
    args = ap.parse_args()

    policy = load_policy(args.policy)   # validates the artifact (fails closed on raw recommendation)
    if args.mode == "raw_rerank" and not args.allow_raw_rerank_dangerous:
        print("ERROR: --mode raw_rerank requires --allow-raw-rerank-dangerous (raw is unsafe)",
              file=sys.stderr)
        return 2

    rows = _read(args.input)
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_override = 0
    try:
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                rec = PR.rerank_query(r, policy, mode=args.mode,
                                      allow_raw=args.allow_raw_rerank_dangerous)
                n_override += int(rec["diagnostics"]["margin_override_used"])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    assert "torch" not in sys.modules, "serving wrapper must not import torch"
    print(f"[policy-rerank] policy={policy.get('policy_id')} mode={args.mode} "
          f"queries={len(rows)} margin_override_used={n_override} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
