#!/usr/bin/env python3
"""Plan or run an MTEB(deu) retrieval-core evaluation as an AutoResearch trial (stdlib planner).

Wraps ``scripts/run_mteb_retrieval_de.py`` reproducibly: --dry-run prints the exact eval command(s)
and writes a trial manifest; --real --allow-gpu runs the eval on the A6000. The fair same-size-peer
comparison is at @512; an optional native-context long-doc pass is planned when enabled.

    python scripts/ar_mteb_trial.py --config configs/autoresearch/mteb_retrieval_core.json \
        --model outputs/v8/diverse-causal/checkpoint --label v8-diverse-causal --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]


def build_mteb_commands(cfg: Dict[str, Any], model: str, label: str,
                        loader: Optional[str] = None) -> Dict[str, Any]:
    tasks = ",".join(cfg.get("tasks", []))
    seq = int(cfg.get("max_seq_length", 512))
    bs = int(cfg.get("batch_size", 32))
    ld = loader or cfg.get("loader", "st")
    def _argv(mdl, lbl, tk, sq):
        return ["python", "scripts/run_mteb_retrieval_de.py", "--model", mdl, "--label", lbl,
                "--tasks", tk, "--loader", ld, "--batch-size", str(bs), "--max-seq-length", str(sq)]

    # argv form is what we EXECUTE (no shell → paths with spaces/metacharacters are safe); the
    # display string keeps the CUDA prefix for the human-readable manifest only.
    argv_commands = [_argv(model, label, tasks, seq)]
    commands = ["CUDA_VISIBLE_DEVICES=0 " + " ".join(argv_commands[0])]
    out = {"label": label, "model": model, "tasks": cfg.get("tasks", []),
           "max_seq_length": seq, "commands": commands, "argv_commands": argv_commands,
           "summary_path": f"outputs/mteb/{label}/summary.json"}
    longdoc = cfg.get("long_doc_native_context", {}) or {}
    if longdoc.get("enabled"):
        lt = ",".join(longdoc.get("tasks", []))
        lseq = int(longdoc.get("max_seq_length", 2048))
        argv_commands.append(_argv(model, f"{label}-{lseq}", lt, lseq))
        commands.append("CUDA_VISIBLE_DEVICES=0 " + " ".join(argv_commands[-1]))
        out["long_doc_summary_path"] = f"outputs/mteb/{label}-{lseq}/summary.json"
        out["long_doc_note"] = (f"the long-doc pass writes a SEPARATE label {label}-{lseq}; the "
                                "promotion gate reads only the primary summary above — pass that "
                                "long-doc label to the gate explicitly if you want it to count")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/autoresearch/mteb_retrieval_core.json")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--loader", default=None)
    ap.add_argument("--out", default="outputs/autoresearch/mteb")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--allow-gpu", action="store_true")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    plan = build_mteb_commands(cfg, args.model, args.label, loader=args.loader)

    out_dir = (Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mteb_trial_manifest.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    if not (args.real and args.allow_gpu):
        print(json.dumps({"status": "dry-run", **plan,
                          "note": "pass --real --allow-gpu to run the eval"},
                         ensure_ascii=False, indent=2))
        return 0
    import os
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    for argv_cmd in plan["argv_commands"]:
        # NO shell: argv list + env, so a model path/label with spaces or shell metacharacters
        # is passed verbatim and can never be interpreted by a shell.
        proc = subprocess.run([sys.executable] + argv_cmd[1:], cwd=str(ROOT), env=env)
        if proc.returncode != 0:
            print(json.dumps({"status": "fail", "failed_command": " ".join(argv_cmd)},
                             ensure_ascii=False))
            return proc.returncode
    print(json.dumps({"status": "ok", "summary_path": plan["summary_path"],
                      "next": f"python scripts/ar_promote.py --candidate {args.label}"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
