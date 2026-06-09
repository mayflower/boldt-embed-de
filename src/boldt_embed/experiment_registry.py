"""Experiment registry: run cards for every teacher-cache / train / eval run (pure stdlib).

A run card is a small JSON provenance record so any number under `outputs/` is traceable to
the exact command, commit, environment, inputs, and outputs that produced it. Versions are
read from package *metadata* (`importlib.metadata`), so collecting env info imports no ML.

Used by the real-run scripts (each takes `--run-id` and calls `emit_run_card`) and summarized
by `scripts/summarize_experiments.py`.
"""
from __future__ import annotations

import importlib.metadata as ilm
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
RUN_CARD_DIR = ROOT / "outputs" / "run-cards"
RUN_TYPES = {"teacher_cache", "train_embedder", "train_reranker", "eval"}
REQUIRED_FIELDS = ("run_id", "run_type", "command", "commit", "date")


def current_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _pkg_version(pkg: str) -> Optional[str]:
    try:
        return ilm.version(pkg)  # reads metadata; does NOT import the package
    except Exception:
        return None


def collect_env_metadata() -> Dict[str, Any]:
    return {
        "commit": current_git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _pkg_version("torch"),
        "transformers": _pkg_version("transformers"),
        "sentence_transformers": _pkg_version("sentence-transformers"),
        "mteb": _pkg_version("mteb"),
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "run"


def default_run_id(run_type: str, commit: Optional[str] = None) -> str:
    commit = (commit or current_git_commit())[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return _slug(f"{run_type}-{commit}-{stamp}")


def new_run_card(run_id: str, run_type: str, command: str, *, model: Optional[str] = None,
                 dataset: Optional[str] = None, metrics: Optional[Dict[str, Any]] = None,
                 input_artifacts: Optional[Sequence[str]] = None,
                 output_artifacts: Optional[Sequence[str]] = None, notes: str = "",
                 gpu: Optional[str] = None, date: Optional[str] = None,
                 env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    env = env or collect_env_metadata()
    return {
        "run_id": run_id,
        "run_type": run_type,
        "command": command,
        "commit": env.get("commit", "unknown"),
        "date": date or _now_iso(),
        "hardware": env.get("platform"),
        "gpu": gpu,
        "python": env.get("python"),
        "torch": env.get("torch"),
        "transformers": env.get("transformers"),
        "sentence_transformers": env.get("sentence_transformers"),
        "input_artifacts": list(input_artifacts or []),
        "output_artifacts": list(output_artifacts or []),
        "model": model,
        "dataset": dataset,
        "metrics": dict(metrics or {}),
        "notes": notes,
    }


def link_artifacts(card: Dict[str, Any], inputs: Optional[Sequence[str]] = None,
                   outputs: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    if inputs:
        card["input_artifacts"] = list(dict.fromkeys(list(card.get("input_artifacts", [])) + list(inputs)))
    if outputs:
        card["output_artifacts"] = list(dict.fromkeys(list(card.get("output_artifacts", [])) + list(outputs)))
    return card


def validate_run_card(card: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(card, dict):
        return ["run card must be a JSON object"]
    for f in REQUIRED_FIELDS:
        if not card.get(f):
            errors.append(f"missing required field '{f}'")
    if card.get("run_type") not in RUN_TYPES:
        errors.append(f"run_type '{card.get('run_type')}' not in {sorted(RUN_TYPES)}")
    if "metrics" in card and not isinstance(card["metrics"], dict):
        errors.append("'metrics' must be an object")
    for f in ("input_artifacts", "output_artifacts"):
        if f in card and not isinstance(card[f], list):
            errors.append(f"'{f}' must be a list")
    return errors


def write_run_card(card: Dict[str, Any], out_dir: Optional[Path] = None) -> str:
    errors = validate_run_card(card)
    if errors:
        raise ValueError("invalid run card: " + "; ".join(errors))
    out_dir = Path(out_dir) if out_dir else RUN_CARD_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_slug(card['run_id'])}.json"
    path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def emit_run_card(run_id: Optional[str], run_type: str, command: str, *,
                  model: Optional[str] = None, dataset: Optional[str] = None,
                  metrics: Optional[Dict[str, Any]] = None,
                  input_artifacts: Optional[Sequence[str]] = None,
                  output_artifacts: Optional[Sequence[str]] = None, notes: str = "",
                  gpu: Optional[str] = None, out_dir: Optional[Path] = None) -> str:
    """Convenience for real-run scripts: build + write a run card, returning its path."""
    env = collect_env_metadata()
    card = new_run_card(run_id or default_run_id(run_type, env.get("commit")), run_type, command,
                        model=model, dataset=dataset, metrics=metrics,
                        input_artifacts=input_artifacts, output_artifacts=output_artifacts,
                        notes=notes, gpu=gpu, env=env)
    return write_run_card(card, out_dir)


def read_run_cards(directory: Optional[Path] = None) -> List[Dict[str, Any]]:
    directory = Path(directory) if directory else RUN_CARD_DIR
    if not directory.exists():
        return []
    cards = []
    for p in sorted(directory.glob("*.json")):
        try:
            cards.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return cards
