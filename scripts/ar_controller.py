#!/usr/bin/env python3
"""AutoResearch state-machine controller — plan / status / next / record (stdlib, no GPU).

The controller is the conservative brain of the v8 frontier program. It reads the event log
(``outputs/autoresearch/events.jsonl``), decides the next trial type with a deterministic ladder
(see ``boldt_embed.autoresearch_state``), and emits a PLAN — the exact command a human/agent would
run next. It does NOT start GPU/teacher work itself: real runs are launched by invoking the trial
script with explicit ``--real``/``--allow-*`` flags. This keeps every expensive step an explicit,
auditable act.

    python scripts/ar_controller.py status
    python scripts/ar_controller.py next --dry-run
    python scripts/ar_controller.py plan --trial-type dense --dry-run
    python scripts/ar_controller.py record --event-json path/to/event.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import autoresearch_state as st  # noqa: E402

DEFAULT_SEARCH_SPACE = ROOT / "configs" / "autoresearch" / "search_space_v8.json"
DEFAULT_OUT = ROOT / "outputs" / "autoresearch" / "controller"


def _load_search_space(path: Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"search space not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def build_plan(trial_type: str, search_space: Dict[str, Any], *,
               real: bool = False, allow_gpu: bool = False, allow_teacher: bool = False,
               allow_checkpoints: bool = False, allow_merge: bool = False) -> Dict[str, Any]:
    """Build a JSON plan (command + artifacts) for one trial type. Pure planning — never executes."""
    if trial_type not in st.TRIAL_TYPES:
        raise SystemExit(f"unknown trial_type {trial_type!r}; expected one of {st.TRIAL_TYPES}")
    spec = (search_space.get("trials", {}) or {}).get(trial_type)
    if not spec:
        raise SystemExit(f"no search-space entry for trial_type {trial_type!r}")

    cmd: List[str] = ["python", spec["script"]]
    if spec.get("config"):
        cmd += ["--config", spec["config"]]
    if spec.get("catalog"):
        cmd += ["--catalog", spec["catalog"]]
    if spec.get("out"):
        cmd += ["--out", spec["out"]]
    if spec.get("out_root"):
        cmd += ["--out-root", spec["out_root"]]

    # Flag policy: dry-run is the default. Real flags are only appended to the PLANNED command when
    # the operator explicitly asked for them — and even then the controller only prints the command,
    # it does not run it (real execution is a separate, explicit step).
    flag_map = {"--real": real, "--allow-gpu": allow_gpu, "--allow-teacher": allow_teacher,
                "--allow-checkpoints": allow_checkpoints, "--allow-merge": allow_merge}
    requested_real = [f for f in (spec.get("real_flags") or []) if flag_map.get(f)]
    if requested_real:
        cmd += requested_real
        mode = "real (planned only — run the command yourself)"
    else:
        cmd += ["--dry-run"]
        mode = "dry-run"

    return {
        "trial_type": trial_type,
        "mode": mode,
        "command": " ".join(cmd),
        "command_argv": cmd,
        "script": spec["script"],
        "config": spec.get("config"),
        "expected_outputs": spec.get("produces", []),
        "available_real_flags": spec.get("real_flags", []),
        "note": "Controller PLANS only; it never starts GPU/teacher work. Run the command to execute.",
    }


def cmd_plan(args, search_space) -> int:
    plan = build_plan(args.trial_type, search_space, real=args.real, allow_gpu=args.allow_gpu,
                      allow_teacher=args.allow_teacher, allow_checkpoints=args.allow_checkpoints,
                      allow_merge=args.allow_merge)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


def cmd_status(args, search_space) -> int:
    events = st.read_events(Path(args.state))
    summary = st.summarize(events)
    summary["state_path"] = str(args.state)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_next(args, search_space) -> int:
    events = st.read_events(Path(args.state))
    nxt = st.decide_next(events)
    if nxt is None:
        print(json.dumps({"next_trial_type": None, "note": "program complete — all ladder steps satisfied"},
                         ensure_ascii=False, indent=2))
        return 0
    plan = build_plan(nxt, search_space, real=args.real, allow_gpu=args.allow_gpu,
                      allow_teacher=args.allow_teacher, allow_checkpoints=args.allow_checkpoints,
                      allow_merge=args.allow_merge)
    plan["decided_by"] = "deterministic ladder over events.jsonl success counts"
    plan["success_counts"] = st.success_counts(events)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


def cmd_record(args, search_space) -> int:
    p = Path(args.event_json)
    if not p.exists():
        raise SystemExit(f"event json not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    # normalize through new_event so the appended record always has the full schema
    event = st.new_event(
        raw.get("trial_type"), raw.get("status", "ok"),
        event_id=raw.get("event_id"), timestamp_utc=raw.get("timestamp_utc"),
        parent_artifacts=raw.get("parent_artifacts"), input_artifacts=raw.get("input_artifacts"),
        output_artifacts=raw.get("output_artifacts"), config=raw.get("config"),
        metrics=raw.get("metrics"), gates=raw.get("gates"), notes=raw.get("notes", ""),
        git=raw.get("git"))
    st.append_event(event, Path(args.state))
    print(json.dumps({"recorded": event["event_id"], "trial_type": event["trial_type"],
                      "status": event["status"], "state_path": str(args.state)},
                     ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["plan", "status", "next", "record"])
    ap.add_argument("--trial-type", choices=list(st.TRIAL_TYPES))
    ap.add_argument("--event-json")
    ap.add_argument("--state", default=str(st.DEFAULT_STATE))
    ap.add_argument("--search-space", default=str(DEFAULT_SEARCH_SPACE))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--dry-run", action="store_true", help="default; explicit for symmetry")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--allow-gpu", action="store_true")
    ap.add_argument("--allow-teacher", action="store_true")
    ap.add_argument("--allow-checkpoints", action="store_true")
    ap.add_argument("--allow-merge", action="store_true")
    args = ap.parse_args(argv)

    search_space = _load_search_space(Path(args.search_space))
    if args.action == "plan":
        if not args.trial_type:
            raise SystemExit("plan requires --trial-type")
        return cmd_plan(args, search_space)
    if args.action == "status":
        return cmd_status(args, search_space)
    if args.action == "next":
        return cmd_next(args, search_space)
    if args.action == "record":
        if not args.event_json:
            raise SystemExit("record requires --event-json")
        return cmd_record(args, search_space)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
