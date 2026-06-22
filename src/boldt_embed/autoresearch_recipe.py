"""Dense-retriever AutoResearch recipe — the only file the AutoResearch loop normally edits.

Two modes:

* **dry-run** (default): pure stdlib. No torch / transformers / sentence-transformers / datasets /
  GPU / network. Validates the config, builds a training plan, derives deterministic metrics
  from the plan + seed, writes ``recipe_plan.json``, and returns a metrics dict matching the
  ``ar_score.py`` schema. Pseudo-metrics are PLUMBING ONLY and carry a ``scale_disclaimer`` — they
  are never a benchmark claim.
* **real**: a safe adapter over the existing v6.1 dense scripts
  (``scripts/train_v6_1_dense_top50.py`` → ``scripts/eval_v6_1_dense_top50.py``). It honors the
  20-minute deadline (stops with a reserve before the budget expires), writes everything inside the
  run directory, and maps the eval summary into the scorer schema. If local data/scripts are
  missing it returns ``status: "fail"`` with the missing integration points — it never fabricates
  metrics.

``torch``/``transformers`` are NEVER imported at module top; the real path imports lazily and the
training/eval themselves run in subprocesses.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

ROOT = Path(__file__).resolve().parents[2]

SCALE_DISCLAIMER = (
    "Dry-run pseudo-metrics validate AutoResearch plumbing only; not a benchmark claim."
)
# Subprocess steps reserve this long (seconds) to write outputs before the budget deadline.
DEFAULT_RESERVE_SECONDS = 30.0


# --------------------------------------------------------------------------- config validation
def validate_recipe_config(config: Dict[str, Any]) -> List[str]:
    """Return a list of problems with the recipe config (empty == ok). Never raises."""
    errors: List[str] = []
    if not isinstance(config, dict):
        return ["config must be a JSON object"]
    if config.get("task", "dense_retriever") != "dense_retriever":
        errors.append("task must be 'dense_retriever' for the dense recipe")
    seed = config.get("seed", 1337)
    if not isinstance(seed, int) or isinstance(seed, bool):
        errors.append("seed must be an integer")
    dims = config.get("matryoshka_dims", [1024, 512, 256, 128])
    if not isinstance(dims, list) or not dims or not all(
        isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in dims
    ):
        errors.append("matryoshka_dims must be a non-empty list of positive ints")
    mix = config.get("data_mixture")
    if mix is not None:
        if not isinstance(mix, dict) or not mix:
            errors.append("data_mixture must be a non-empty object when present")
        else:
            bad = [k for k, v in mix.items()
                   if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0]
            if bad:
                errors.append(f"data_mixture weights must be non-negative numbers: {sorted(bad)}")
            elif abs(sum(mix.values()) - 1.0) > 1e-6:
                errors.append(f"data_mixture weights must sum to 1.0 (got {sum(mix.values())})")
    loss = config.get("loss", {})
    if loss and not isinstance(loss.get("type", ""), str):
        errors.append("loss.type must be a string")
    return errors


# ------------------------------------------------------------------------------- training plan
def build_training_plan(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the search-relevant fields into a flat, JSON-serializable plan.

    The plan is what dry-run pseudo-metrics are derived from, so it must contain exactly the fields
    a real trial would actually vary (pooling, dims, mixture, loss, optimizer)."""
    training = config.get("training", {}) or {}
    loss = config.get("loss", {}) or {}
    return {
        "task": config.get("task", "dense_retriever"),
        "base_model": config.get("base_model", "Boldt/Boldt-DC-350M"),
        "reference_model": config.get("reference_model"),
        "seed": int(config.get("seed", 1337)),
        "pooling": config.get("pooling", "mean"),
        "normalize_embeddings": bool(config.get("normalize_embeddings", True)),
        "matryoshka_dims": list(config.get("matryoshka_dims", [1024, 512, 256, 128])),
        "data_mixture": dict(config.get("data_mixture", {})),
        "loss_type": loss.get("type", "cached_mnrl_matryoshka_distillation"),
        "temperature": loss.get("temperature", 0.03),
        "matryoshka_weight": loss.get("matryoshka_weight", 1.0),
        "distillation_weight": loss.get("distillation_weight", 0.5),
        "margin_mse_weight": loss.get("margin_mse_weight", 0.25),
        "learning_rate": training.get("learning_rate", 2e-5),
        "warmup_ratio": training.get("warmup_ratio", 0.05),
        "batch_size": training.get("batch_size", 32),
        "grad_accumulation": training.get("grad_accumulation", 1),
        "max_query_length": training.get("max_query_length", 256),
        "max_document_length": training.get("max_document_length", 1024),
        # The trainer applies ONE max_seq_length to the whole encoder, so use the larger of the
        # two so documents are not silently truncated to the (shorter) query length.
        "max_seq_length": max(int(training.get("max_query_length", 256)),
                              int(training.get("max_document_length", 1024))),
        "max_steps": training.get("max_steps"),
        "dtype": training.get("dtype", "bfloat16"),
    }


# --------------------------------------------------------------------------------- deadline
def should_stop(deadline_epoch_s: float, reserve_seconds: float = DEFAULT_RESERVE_SECONDS) -> bool:
    """True when there is no longer enough time (minus a reserve) to start more work."""
    return time.monotonic() >= (deadline_epoch_s - reserve_seconds)


def _remaining_seconds(deadline_epoch_s: float) -> float:
    return max(0.0, deadline_epoch_s - time.monotonic())


# ----------------------------------------------------------------------------------- metrics IO
def _metrics_skeleton(run_id: str, status: str) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "status": status,
        "metrics": {
            "webfaq": {"recall@100": 0.0, "ndcg@10": 0.0, "mrr@10": 0.0},
            "germanquad": {"ndcg@10": 0.0},
            "dt_test": {"ndcg@10": 0.0},
            "matryoshka": {"retention_256": 0.0},
            "leakage": {"hits": 0},
            "system": {"vram_gb": 0.0, "throughput_pairs_per_sec": 0.0},
        },
    }


def write_metrics(out_dir: Union[str, Path], metrics: Dict[str, Any]) -> None:
    """Write ``metrics.json`` into the run directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ------------------------------------------------------------------------------- pseudo-metrics
def _frac(digest: bytes, slot: int) -> float:
    """Deterministic float in [0, 1) from a 4-byte slice of a 32-byte digest (slots 0..7)."""
    start = (slot * 4) % len(digest)
    return int.from_bytes(digest[start:start + 4], "big") / float(1 << 32)


def _band(frac: float, lo: float, hi: float, ndigits: int = 4) -> float:
    return round(lo + frac * (hi - lo), ndigits)


def _pseudo_metrics(plan: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    """Deterministic, schema-compatible pseudo-metrics derived from the plan + seed.

    Same plan + seed → identical numbers; any search-relevant config change → different numbers.
    These are plumbing checks, NOT model quality — hence the scale_disclaimer in the caller."""
    canonical = json.dumps(plan, sort_keys=True, ensure_ascii=False) + f"|seed={plan['seed']}"
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return {
        "webfaq": {
            "recall@100": _band(_frac(digest, 0), 0.90, 0.98),
            "ndcg@10": _band(_frac(digest, 1), 0.60, 0.72),
            "mrr@10": _band(_frac(digest, 2), 0.55, 0.68),
        },
        "germanquad": {"ndcg@10": _band(_frac(digest, 3), 0.86, 0.92)},
        "dt_test": {"ndcg@10": _band(_frac(digest, 4), 0.93, 0.98)},
        # band kept >= the 0.95 gate floor so plumbing dry-runs don't spuriously trip the
        # retention gate (a dry-run is non-promotable regardless — see ar_score mode gate).
        "matryoshka": {"retention_256": _band(_frac(digest, 5), 0.95, 0.99)},
        "leakage": {"hits": 0, "status": "not_checked"},
        "system": {
            "vram_gb": _band(_frac(digest, 6), 10.0, 24.0, 2),
            "throughput_pairs_per_sec": _band(_frac(digest, 7), 200.0, 1200.0, 1),
        },
    }


# ------------------------------------------------------------------------------------ dry-run
def _run_dry(config: Dict[str, Any], out_dir: Path, run_id: str) -> Dict[str, Any]:
    errors = validate_recipe_config(config)
    plan = build_training_plan(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "recipe_plan.json").write_text(
        json.dumps({"mode": "dry_run", "config_errors": errors, "training_plan": plan},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        result = _metrics_skeleton(run_id, "fail")
        result["mode"] = "dry_run"
        result["note"] = "config invalid: " + "; ".join(errors)
        result["scale_disclaimer"] = SCALE_DISCLAIMER
        return result
    result = {
        "run_id": run_id,
        "status": "ok",
        "mode": "dry_run",
        "scale_disclaimer": SCALE_DISCLAIMER,
        "metrics": _pseudo_metrics(plan, run_id),
        "training_plan": plan,
    }
    return result


# --------------------------------------------------------------------------------------- real
def _load_eval_module():
    """Load the (protected) eval script as a module so we can reuse its metric/eval functions
    without editing or subprocessing it. Its top-level imports are stdlib + boldt_embed.metrics
    (no torch at import time); torch is imported lazily inside dense_eval()."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eval_v6_1_dense_top50", ROOT / "scripts" / "eval_v6_1_dense_top50.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _looks_local(p: str) -> bool:
    """A model ref is a local path (must exist) rather than a remote HF id."""
    return p.startswith(("./", "/", "outputs/", "data/")) or (ROOT / p).exists()


def _resolve_model_path(p: str) -> str:
    """Make a local repo checkpoint path absolute; leave a remote HF id untouched."""
    if p.startswith("/"):
        return p
    if p.startswith(("./", "outputs/", "data/")):
        return str((ROOT / p).resolve())
    return p


def _leakage_from_manifest(runtime: Dict[str, Any]) -> Dict[str, Any]:
    """Leakage is a property of the data PREPARATION, not the eval. Pull it from the prepared
    manifest (if supplied) so the scorer gates on a real, verified status — never fabricated."""
    mp = runtime.get("prepared_manifest")
    if not mp:
        return {"hits": None, "status": "not_checked"}
    p = Path(mp) if Path(mp).is_absolute() else (ROOT / mp)
    try:
        man = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"hits": None, "status": "unreadable"}
    lk = man.get("leakage") or {}
    return {"hits": lk.get("hits"), "status": lk.get("status", "unknown")}


def _evaluate_in_process(ev, label: str, model_spec: Dict[str, Any], eval_sets: List[str],
                         eval_set_paths: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate ONE checkpoint over the eval sets by calling the eval script's own functions,
    so we grade exactly the checkpoint we were given (no global-path indirection)."""
    summary: Dict[str, Any] = {label: {}}
    for s in eval_sets:
        corpus_p, queries_p, qrels_p, role = eval_set_paths[s]
        corpus = ev._read(str(ROOT / corpus_p))
        queries = ev._read(str(ROOT / queries_p))
        qrels = ev._qrels(str(ROOT / qrels_p), queries)
        if model_spec.get("kind") == "bm25":
            res = ev.bm25_eval(corpus, queries, qrels)
        else:
            res = ev.dense_eval(
                model_spec["path"], corpus, queries, qrels,
                query_prefix=model_spec.get("query_prefix", ""),
                doc_prefix=model_spec.get("doc_prefix", ""),
                matryoshka=model_spec.get("matryoshka", True),
                trust_remote_code=model_spec.get("trust_remote_code", False))
        res["model"], res["eval_set"], res["role"] = label, s, role
        summary[label][s] = res
    return summary


def _gpu_vram_gb() -> float:
    try:
        import torch  # lazy

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
    except Exception:
        pass
    return 0.0


def _map_eval_summary(summary: Dict[str, Any], model_name: str,
                      leakage: Dict[str, Any]) -> Dict[str, Any]:
    """Map an eval summary {model:{set:metrics}} into the scorer schema for ``model_name``.
    ``leakage`` carries the verified status from the prepared manifest — it is NOT fabricated."""
    by_set = summary.get(model_name) or {}
    metrics: Dict[str, Any] = {}
    wf = by_set.get("webfaq") or {}
    metrics["webfaq"] = {
        "recall@100": wf.get("recall@100"),
        "ndcg@10": wf.get("ndcg@10"),
        "mrr@10": wf.get("mrr@10"),
    }
    gq = by_set.get("germanquad") or {}
    metrics["germanquad"] = {"ndcg@10": gq.get("ndcg@10")}
    dt = by_set.get("dt_test") or {}
    metrics["dt_test"] = {"ndcg@10": dt.get("ndcg@10")}
    if "local_rag" in by_set:
        lr = by_set["local_rag"]
        metrics["local_rag"] = {"recall@100": lr.get("recall@100"), "ndcg@10": lr.get("ndcg@10")}
    metrics["matryoshka"] = {"retention_256": wf.get("matryoshka_256_retention")}
    metrics["leakage"] = dict(leakage)
    tput = (wf.get("throughput") or {}).get("queries_per_sec", 0.0)
    metrics["system"] = {"vram_gb": _gpu_vram_gb(), "throughput_pairs_per_sec": tput}
    return metrics


def _subprocess(cmd: List[str], *, cwd: Path, env: Dict[str, str], timeout: float,
                log: List[Dict[str, Any]]) -> subprocess.CompletedProcess:
    log.append({"cmd": cmd, "timeout_s": round(timeout, 1)})
    return subprocess.run(cmd, cwd=str(cwd), env=env, timeout=timeout,
                          capture_output=True, text=True)


def _run_real(config: Dict[str, Any], out_dir: Path, run_id: str,
              deadline_epoch_s: float) -> Dict[str, Any]:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = build_training_plan(config)
    runtime = config.get("runtime", {}) or {}
    cmd_log: List[Dict[str, Any]] = []

    def fail(note: str, **extra: Any) -> Dict[str, Any]:
        res = _metrics_skeleton(run_id, "fail")
        res.update({"mode": "real", "note": note, "training_plan": plan})
        res.update(extra)
        (out_dir / "recipe_commands.json").write_text(
            json.dumps({"commands": cmd_log, "note": note}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        return res

    errors = validate_recipe_config(config)
    if errors:
        return fail("config invalid: " + "; ".join(errors), config_errors=errors)

    train_script = ROOT / "scripts" / "train_v6_1_dense_top50.py"
    eval_script = ROOT / "scripts" / "eval_v6_1_dense_top50.py"
    if not eval_script.exists():
        raise NotImplementedError(
            f"real mode needs {eval_script.relative_to(ROOT)}; integration point missing")
    try:
        ev = _load_eval_module()
    except Exception as exc:  # the eval script must be importable
        return fail(f"could not load eval module: {exc}")
    eval_set_paths = dict(getattr(ev, "EVAL_SETS", {}))

    env = dict(os.environ)
    cvd = runtime.get("cuda_visible_devices")
    if cvd is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cvd)

    # --- discover required local inputs (no downloads) ---
    raw_sets = str(runtime.get("eval_sets", "webfaq,germanquad,dt_test"))
    eval_sets = [s.strip() for s in raw_sets.split(",") if s.strip()]
    missing: List[str] = []
    for s in eval_sets:
        spec = eval_set_paths.get(s)
        if not spec:
            missing.append(f"eval-set:{s} (unknown)")
            continue
        for p in spec[:3]:  # corpus, queries, qrels
            if not (ROOT / p).exists():
                missing.append(p)

    do_train = bool(runtime.get("train", True))
    train_pairs = runtime.get("train_pairs")
    hard_negs = runtime.get("hard_negatives")
    train_base = runtime.get("train_base_model")
    if do_train:
        if not train_script.exists():
            missing.append(f"{train_script.relative_to(ROOT)} (train script)")
        for p in (train_pairs, hard_negs):
            if not p or not (ROOT / p).exists():
                missing.append(str(p))
        if not train_base:
            missing.append("train_base_model (required when training)")
        elif _looks_local(str(train_base)) and not (ROOT / str(train_base)).exists():
            missing.append(str(train_base))

    if missing:
        return fail(
            "missing local integration inputs (no benchmark claim possible) — supply these files "
            "or run with --dry-run", missing_inputs=sorted(set(missing)))

    if should_stop(deadline_epoch_s, DEFAULT_RESERVE_SECONDS):
        return fail("insufficient remaining budget to start a real trial")

    leakage = _leakage_from_manifest(runtime)

    # --- decide which checkpoint to grade, training to the RUN DIR (never the promoted path) ---
    if do_train:
        ckpt = out_dir / "checkpoint"
        eval_reserve = float(runtime.get("eval_reserve_seconds", 300.0))
        timeout = _remaining_seconds(deadline_epoch_s) - eval_reserve
        if timeout <= 0:
            return fail(f"insufficient budget to train and still evaluate "
                        f"(need > {eval_reserve:.0f}s reserved for eval)")
        steps = plan.get("max_steps") or int(runtime.get("max_steps", 1000))
        train_cmd = [sys.executable, str(train_script),
                     "--base-model", str(train_base),
                     "--train-pairs", str(train_pairs),
                     "--hard-negatives", str(hard_negs),
                     "--output", str(ckpt),
                     "--max-steps", str(steps),
                     "--batch-size", str(plan["batch_size"]),
                     "--max-seq-length", str(plan["max_seq_length"]),
                     "--run-id", f"{run_id}-train"]
        if plan["dtype"] == "bfloat16":
            train_cmd.append("--bf16")
        try:
            proc = _subprocess(train_cmd, cwd=ROOT, env=env, timeout=timeout, log=cmd_log)
        except subprocess.TimeoutExpired:
            return fail("training exceeded its budget slice (deadline)", deadline_respected=False)
        if proc.returncode != 0:
            (out_dir / "train.stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
            return fail(f"training failed (exit {proc.returncode}); see train.stderr.txt")
        label = "dense-trial"
        model_spec = {"kind": "dense", "path": str(ckpt), "matryoshka": True}
    else:
        label = runtime.get("eval_model_name", "dense-v6.1")
        spec = dict(getattr(ev, "MODEL_SPECS", {})).get(label)
        if not spec:
            return fail(f"unknown eval_model_name {label!r} (not in eval MODEL_SPECS)")
        model_spec = dict(spec)
        path = model_spec.get("path")
        if path:
            if model_spec.get("kind") == "dense" and path.startswith("outputs") \
                    and not (ROOT / path).exists():
                return fail(f"checkpoint for {label!r} missing: {path}")
            model_spec["path"] = _resolve_model_path(path)

    if should_stop(deadline_epoch_s, 5.0):
        return fail("insufficient remaining budget before evaluation")

    # --- evaluation: reuse the eval script's own functions, grading exactly this checkpoint ---
    try:
        summary = _evaluate_in_process(ev, label, model_spec, eval_sets, eval_set_paths)
    except Exception as exc:
        (out_dir / "eval.error.txt").write_text(repr(exc), encoding="utf-8")
        return fail(f"evaluation failed: {type(exc).__name__}: {exc}")

    summary_path = out_dir / "dense_eval_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = _map_eval_summary(summary, label, leakage)
    (out_dir / "recipe_commands.json").write_text(
        json.dumps({"commands": cmd_log}, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {
        "run_id": run_id,
        "status": "ok" if metrics["webfaq"].get("recall@100") is not None else "fail",
        "mode": "real",
        "metrics": metrics,
        "training_plan": plan,
        "eval_model": label,
        "eval_summary_path": str(summary_path),
        "leakage_status": leakage.get("status"),
        "trained": do_train,
    }
    if result["status"] == "fail":
        result["note"] = f"eval summary lacked WebFAQ metrics for {label!r}"
    return result


# ------------------------------------------------------------------------------- public entry
def run_dense_trial(config: Dict[str, Any], out_dir: Union[str, Path],
                    deadline_epoch_s: float, dry_run: bool = True) -> Dict[str, Any]:
    """Run one dense-retriever AutoResearch trial and return a metrics dict (scorer-compatible)."""
    out = Path(out_dir)
    run_id = str(config.get("run_id") or config.get("name") or out.name)
    if dry_run:
        return _run_dry(config, out, run_id)
    return _run_real(config, out, run_id, deadline_epoch_s)
