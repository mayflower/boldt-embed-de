"""AutoResearch state machine — stdlib-only event log + deterministic next-trial decision.

The orchestrator is *stateful*: every planned/run/observed trial appends one JSON event to
``outputs/autoresearch/events.jsonl``. The controller (``scripts/ar_controller.py``) reads the log,
decides the next trial type with a deterministic ladder, and emits a plan — it NEVER starts GPU work
on its own. No ML imports here (pure stdlib), so status/plan/next run with no torch/weights/GPU.

Trial types form the v8 program:
    data_mix -> dense -> hardneg_refresh -> specialist (x2) -> merge -> distill -> mteb -> promotion
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = ROOT / "outputs" / "autoresearch" / "events.jsonl"

TRIAL_TYPES = (
    "data_mix", "dense", "hardneg_refresh", "specialist",
    "merge", "distill", "mteb", "promotion",
)
STATUSES = ("planned", "running", "ok", "fail", "skipped")

# The deterministic ladder: (trial_type, predicate over the success-counts dict -> needs this step).
# Evaluated top-down; the first step whose predicate is True is the next trial.
_LADDER = (
    ("data_mix", lambda n: n["data_mix"] == 0),
    ("dense", lambda n: n["dense"] == 0),
    ("hardneg_refresh", lambda n: n["hardneg_refresh"] == 0),
    ("specialist", lambda n: n["specialist"] < 2),
    ("merge", lambda n: n["merge"] == 0),
    ("distill", lambda n: n["distill"] == 0),
    ("mteb", lambda n: n["mteb"] == 0),
    ("promotion", lambda n: n["promotion"] == 0),
)


def now_iso(now: Optional[_dt.datetime] = None) -> str:
    """UTC ISO timestamp. Injectable so tests are deterministic."""
    now = now or _dt.datetime.now(_dt.timezone.utc)
    return now.isoformat()


def git_info() -> Dict[str, Any]:
    """Best-effort {commit, dirty}; never raises (returns unknown on any failure)."""
    def _run(args: Sequence[str]) -> Optional[str]:
        try:
            out = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True,
                                 text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None
    commit = _run(["rev-parse", "HEAD"])
    status = _run(["status", "--porcelain"])
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def new_event(trial_type: str, status: str = "planned", *,
              event_id: Optional[str] = None,
              timestamp_utc: Optional[str] = None,
              parent_artifacts: Optional[List[str]] = None,
              input_artifacts: Optional[List[str]] = None,
              output_artifacts: Optional[List[str]] = None,
              config: Optional[Dict[str, Any]] = None,
              metrics: Optional[Dict[str, Any]] = None,
              gates: Optional[Dict[str, Any]] = None,
              notes: str = "",
              git: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build one canonical event dict. event_id/timestamp/git are injectable for tests."""
    if trial_type not in TRIAL_TYPES:
        raise ValueError(f"unknown trial_type {trial_type!r}; expected one of {TRIAL_TYPES}")
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}; expected one of {STATUSES}")
    return {
        "event_id": event_id or uuid.uuid4().hex[:16],
        "timestamp_utc": timestamp_utc or now_iso(),
        "trial_type": trial_type,
        "status": status,
        "parent_artifacts": list(parent_artifacts or []),
        "input_artifacts": list(input_artifacts or []),
        "output_artifacts": list(output_artifacts or []),
        "config": dict(config or {}),
        "metrics": dict(metrics or {}),
        "gates": dict(gates or {}),
        "notes": notes,
        "git": dict(git) if git is not None else git_info(),
    }


def append_event(event: Dict[str, Any], state_path: Path = DEFAULT_STATE) -> Path:
    """Append one event as a JSONL line (creates parent dirs). Returns the state path."""
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    return state_path


def read_events(state_path: Path = DEFAULT_STATE) -> List[Dict[str, Any]]:
    """Read all events (skips blank/corrupt lines rather than crashing)."""
    state_path = Path(state_path)
    if not state_path.exists():
        return []
    events: List[Dict[str, Any]] = []
    for line in state_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def success_counts(events: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Count successful (status==ok) events per trial type."""
    counts = {t: 0 for t in TRIAL_TYPES}
    for e in events:
        if e.get("status") == "ok" and e.get("trial_type") in counts:
            counts[e["trial_type"]] += 1
    return counts


def decide_next(events: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Deterministic next trial type from the success-counts ladder; None when the program is done."""
    n = success_counts(events)
    for trial_type, needs in _LADDER:
        if needs(n):
            return trial_type
    return None


def summarize(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Machine-readable state summary for `ar_controller.py status`."""
    by_type_status: Dict[str, Dict[str, int]] = {}
    for e in events:
        t = e.get("trial_type", "?")
        s = e.get("status", "?")
        by_type_status.setdefault(t, {}).setdefault(s, 0)
        by_type_status[t][s] += 1
    return {
        "n_events": len(events),
        "success_counts": success_counts(events),
        "by_type_status": by_type_status,
        "next_trial_type": decide_next(events),
        "last_events": [
            {k: e.get(k) for k in ("event_id", "timestamp_utc", "trial_type", "status", "notes")}
            for e in events[-5:]
        ],
    }
