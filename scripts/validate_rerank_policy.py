#!/usr/bin/env python3
"""Validate a rerank-policy artifact (stdlib, no ML). Checks the schema + the hard guards (no raw
always-rerank recommendation; no forbidden inference feature in the allowed set; bounds + validation
thresholds present), and optionally that the pinned checkpoint exists. Prints JSON or Markdown.
Exit non-zero on any failure.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import policy_config as PC  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--format", choices=["json", "markdown"], default="json")
    ap.add_argument("--require-model-exists", action="store_true",
                    help="fail (not warn) if the pinned checkpoint is absent on disk")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()

    p = pathlib.Path(args.policy)
    if not p.exists():
        print(f"ERROR: policy not found: {p}", file=sys.stderr)
        return 2
    d = json.loads(p.read_text(encoding="utf-8"))
    errors = PC.validate_policy(d)
    model_ok, model_msg = PC.check_model_exists(d, root=args.root, require=args.require_model_exists)
    if not model_ok:
        errors = errors + [model_msg]
    warnings = [model_msg] if (model_ok and model_msg and model_msg.startswith("WARNING")) else []

    status = "pass" if not errors else "fail"
    report = {"status": status, "policy_id": d.get("policy_id"),
              "policy_type": d.get("policy_type"),
              "raw_always_rerank_recommended": d.get("raw_always_rerank_recommended"),
              "recommended_mode": d.get("recommended_mode"),
              "model_checkpoint": d.get("model_checkpoint"),
              "errors": errors, "warnings": warnings}

    if args.format == "markdown":
        L = [f"# Rerank policy validation: **{status}**", "",
             f"- policy_id: `{d.get('policy_id')}`  type: `{d.get('policy_type')}`",
             f"- recommended_mode: `{d.get('recommended_mode')}`  "
             f"raw_always_rerank_recommended: `{d.get('raw_always_rerank_recommended')}`",
             f"- model_checkpoint: `{d.get('model_checkpoint')}`", ""]
        L.append("## Errors\n" + ("\n".join(f"- ❌ {e}" for e in errors) if errors else "none"))
        if warnings:
            L.append("\n## Warnings\n" + "\n".join(f"- ⚠️ {w}" for w in warnings))
        print("\n".join(L))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))

    if status != "pass":
        print(f"FAIL — {len(errors)} problem(s) in the rerank policy", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
