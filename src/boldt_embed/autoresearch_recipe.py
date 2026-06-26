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
from typing import Any, Dict, List, Optional, Tuple, Union

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
    # Catalogue hygiene: when the mixture will be MATERIALIZED into a real training corpus, every
    # source id must resolve in configs/data_sources.json, be training_usable, and be leakage-clean.
    # (For a non-materialized mix the weights only steer dry-run pseudo-metrics, so we don't gate it.)
    runtime = config.get("runtime", {}) or {}
    if runtime.get("materialize_mixture") and isinstance(mix, dict) and mix:
        catalogue = _load_catalogue()
        for sid, weight in mix.items():
            rec = catalogue.get(sid)
            if rec is None:
                errors.append(f"data_mixture source {sid!r} is not in configs/data_sources.json "
                              "(materialize_mixture=true requires a catalogued source)")
                continue
            if not rec.get("training_usable"):
                errors.append(f"data_mixture source {sid!r} is training_usable=false "
                              "(not allowed in a materialized mixture)")
            elif rec.get("leakage") not in ("scanned_clean", "clean"):
                errors.append(f"data_mixture source {sid!r} leakage={rec.get('leakage')!r} — only "
                              "scanned_clean/clean sources may be materialized "
                              "(run scripts/run_full_leakage_scan.py first)")
            if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0:
                errors.append(f"data_mixture source {sid!r} weight must be > 0 when materialized")
    return errors


# ------------------------------------------------------------------------------- training plan
def build_training_plan(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the search-relevant fields into a flat, JSON-serializable plan.

    The plan is what dry-run pseudo-metrics are derived from, so it must contain exactly the fields
    a real trial would actually vary (pooling, dims, mixture, loss, optimizer)."""
    training = config.get("training", {}) or {}
    loss = config.get("loss", {}) or {}
    batch_size = int(training.get("batch_size", 32))
    max_query = int(training.get("max_query_length", 256))
    max_doc = int(training.get("max_document_length", 256))
    # The trainer applies ONE max_seq_length to the whole encoder, so we'd use the larger of
    # query/doc to avoid truncating documents to the (shorter) query length. BUT batch_size ×
    # seq_length drives activation memory: 32 × 1024 OOMs the 48 GB A6000 (~45 GB peak). Cap the
    # effective seq so batch × seq never exceeds the v6.1-proven operating point (256 × 32 tokens),
    # and never below the query length (queries must not truncate). A config that wants longer
    # documents must "buy" the length by lowering batch_size. Both values are recorded so the cap
    # is visible in the run card rather than silent.
    SAFE_BATCH_TOKENS = 256 * 32
    requested_seq = max(max_query, max_doc)
    seq_cap = max(max_query, SAFE_BATCH_TOKENS // max(batch_size, 1))
    effective_seq = min(requested_seq, seq_cap)
    grad_accum = max(1, int(training.get("grad_accumulation", 1)))
    mini_batch = training.get("mini_batch_size")
    # effective contrastive batch = per-device batch × accumulation steps. An explicit
    # effective_batch_size in the config is honored only when consistent; otherwise the derived value
    # wins and the inconsistency is recorded (never silently trusted).
    derived_eff_batch = batch_size * grad_accum
    requested_eff_batch = training.get("effective_batch_size")
    temp_schedule = training.get("temperature_schedule", "constant")
    grad_ckpt = bool(training.get("gradient_checkpointing", False))
    max_triplets = training.get("max_triplets_per_query")
    # Honesty: which knobs actually reach REAL training vs. are plan-only (so the run card never
    # implies an inert knob was active). The trainer's CMNRL scale is constant, so any non-constant
    # temperature_schedule is plan-only until a real scheduler exists.
    plan_only = []
    if temp_schedule not in (None, "constant"):
        plan_only.append("temperature_schedule")
    if requested_eff_batch is not None and requested_eff_batch != derived_eff_batch:
        plan_only.append("effective_batch_size(requested≠batch×accum)")
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
        "batch_size": batch_size,
        "grad_accumulation": grad_accum,
        "mini_batch_size": int(mini_batch) if mini_batch else None,
        "effective_batch_size": derived_eff_batch,
        "effective_batch_size_requested": requested_eff_batch,
        "gradient_checkpointing": grad_ckpt,
        "temperature_schedule": temp_schedule,
        "max_triplets_per_query": max_triplets,
        "max_query_length": max_query,
        "max_document_length": max_doc,
        "max_seq_length": effective_seq,
        "max_seq_length_requested": requested_seq,
        "seq_capped_for_memory": effective_seq < requested_seq,
        "max_steps": training.get("max_steps"),
        "dtype": training.get("dtype", "bfloat16"),
        "plan_only_knobs": plan_only,
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
    status = lk.get("status", "unknown")
    hits = lk.get("hits")
    # A 'clean' manifest means the TRAINED data is verified clean — either zero hits, or the
    # flagged rows were dropped to a cleaned candidate file that training actually uses. The
    # manifest's `hits` is the RAW pre-clean count (provenance); the effective leakage of the
    # trained data is 0, so report 0 so the scorer's fail-closed gate passes on verified-clean data.
    if status == "clean":
        hits = 0
    return {"hits": hits, "status": status}


def _clean_train_pairs_from_manifest(runtime: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """When a CLEAN prepared manifest is supplied, return the verified-clean training file the
    leakage gate certified, so REAL training provably uses the SAME data the gate passed on (not
    the raw, possibly-leaky ``runtime.train_pairs`` inherited from the base config).

    Returns ``(clean_path | None, error | None)``:
      - clean path  → train on this instead of the configured pairs;
      - (None,None) → no manifest, not verified clean, or genuinely zero-hit (use configured pairs);
      - (None,err)  → FAIL CLOSED: the manifest claims clean but its cleaned file can't be located —
                      we never train on raw data while reporting "leakage clean".
    """
    mp = runtime.get("prepared_manifest")
    if not mp:
        return None, None
    p = Path(mp) if Path(mp).is_absolute() else (ROOT / mp)
    try:
        man = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None, None  # the leakage gate already reports 'unreadable' → not promotable
    lk = man.get("leakage") or {}
    if lk.get("status") != "clean":
        return None, None  # not verified clean → use configured data (won't be promotable)
    # Locate the scan's certified cleaned-candidates file (the exact file the gate verified).
    rep = lk.get("report") or {}
    ccp = None
    rep_path = rep.get("path")
    if rep_path:
        rp = Path(rep_path) if Path(rep_path).is_absolute() else (ROOT / rep_path)
        try:
            ccp = json.loads(rp.read_text(encoding="utf-8")).get("cleaned_candidates_path")
        except Exception:
            ccp = None
    if not ccp:  # fall back to a summary embedded directly in the manifest
        ccp = (rep.get("summary") or {}).get("cleaned_candidates_path")
    if ccp:
        cpath = Path(ccp) if Path(ccp).is_absolute() else (ROOT / ccp)
        if cpath.exists():
            return ccp, None
        return None, f"prepared manifest is clean but its cleaned file is missing: {ccp}"
    # 'clean' with no cleaned file is only valid when there was genuinely nothing to drop.
    if lk.get("hits") in (0, None):
        return None, None
    return None, ("prepared manifest reports clean but records neither a cleaned_candidates_path "
                  "nor zero raw hits — refusing to train on unverified data")


_CATALOGUE_PATH = ROOT / "configs" / "data_sources.json"


def _load_catalogue() -> Dict[str, Dict[str, Any]]:
    """Map every catalogue source id -> its record (path/leakage/training_usable)."""
    out: Dict[str, Dict[str, Any]] = {}
    try:
        cat = json.loads(_CATALOGUE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return out
    for grp in ("train_pairs_processed_unions", "train_pairs_raw_sources"):
        for s in cat.get(grp, []) or []:
            if isinstance(s, dict) and s.get("id"):
                out[s["id"]] = s
    return out


def _materialize_data_mixture(config: Dict[str, Any], out_dir: Path,
                              cmd_log: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """Build ONE real training corpus from catalogue sources named in ``config['data_mixture']``
    (weights, keys = `configs/data_sources.json` ids), so the loop can search data MIXES in REAL
    mode (data_mixture otherwise only steers dry-run pseudo-metrics).

    The actual build is DELEGATED to ``boldt_embed.data_mixture_optimizer.build_mixture`` — the single
    canonical mixture builder — so the fail-closed leakage/training_usable gate, stride-sampling,
    dedup and FAQ cap can never drift between this path and ``scripts/ar_build_mixture.py``. Only
    sources flagged ``training_usable`` AND ``leakage in {scanned_clean, clean}`` are admitted (the
    union of individually-scanned-clean sources is clean, so no in-trial re-scan). Opt-in:
    runtime.materialize_mixture must be true. Returns (train_jsonl_path | None, error | None).
    Pure stdlib (no torch)."""
    runtime = config.get("runtime", {}) or {}
    if not runtime.get("materialize_mixture"):
        return None, None
    mixture = config.get("data_mixture") or {}
    if not mixture:
        return None, "materialize_mixture set but data_mixture is empty"
    total = int(runtime.get("mixture_total", 500000))
    faq_cap = float(runtime.get("faq_cap", 0.30))
    # translate the recipe's compact config into the optimizer's config shape, then delegate
    opt_config = {
        "name": "ar_materialized_mixture",
        "total_rows": total,
        "sources": {sid: float(w) for sid, w in mixture.items()},
        "constraints": {"faq_cap": faq_cap},
    }
    try:
        from boldt_embed import data_mixture_optimizer as dmo  # stdlib, lazy
        catalogue = dmo.load_catalogue(_CATALOGUE_PATH)
        result = dmo.build_mixture(opt_config, catalogue, out_dir=Path(out_dir))
    except Exception as exc:  # MixtureConfigError (fail-closed, names the source id) or IO error
        return None, f"{type(exc).__name__}: {exc}"
    train_path = result.get("written", {}).get("train")
    if not train_path:
        return None, "data_mixture produced no usable examples"
    manifest = result.get("manifest", {})
    cmd_log.append({"note": "materialized data_mixture via data_mixture_optimizer "
                            "(single builder; scanned_clean sources only)",
                    "data_mixture_sources": manifest.get("sources"), "faq_cap": faq_cap,
                    "mixed_rows": manifest.get("rows_written"),
                    "domain_mix": manifest.get("domain_mix"),
                    "leakage": manifest.get("leakage"), "mixed_path": train_path})
    return str(train_path), None


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


def _build_train_cmd(train_script, *, train_base, train_pairs, hard_negs, ckpt, steps, plan,
                     run_id) -> List[str]:
    """Build the v6.1 training subprocess command, forwarding the tunable knobs from the plan.

    lr / warmup_ratio / temperature now reach REAL training (not just the dry-run plan); the
    contrastive batch size = ``batch_size`` (cached MNRL uses every in-batch item as a negative).
    """
    cmd = [sys.executable, str(train_script),
           "--base-model", str(train_base),
           "--train-pairs", str(train_pairs),
           "--hard-negatives", str(hard_negs),
           "--output", str(ckpt),
           "--max-steps", str(steps),
           "--batch-size", str(plan["batch_size"]),
           "--max-seq-length", str(plan["max_seq_length"]),
           "--lr", str(plan["learning_rate"]),
           "--warmup-ratio", str(plan["warmup_ratio"]),
           "--run-id", f"{run_id}-train"]
    if plan.get("temperature") is not None:
        cmd += ["--temperature", str(plan["temperature"])]
    # generalization knobs (Prompt 06) — only forward what the trainer really supports
    if int(plan.get("grad_accumulation", 1)) > 1:
        cmd += ["--grad-accumulation", str(plan["grad_accumulation"])]
    if plan.get("mini_batch_size"):
        cmd += ["--mini-batch-size", str(plan["mini_batch_size"])]
    if plan.get("max_triplets_per_query"):
        cmd += ["--max-triplets-per-query", str(plan["max_triplets_per_query"])]
    if plan.get("gradient_checkpointing"):
        cmd.append("--gradient-checkpointing")
    if plan.get("dtype") == "bfloat16":
        cmd.append("--bf16")
    return cmd


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
    # Data-mixture (opt-in): build a real training corpus from catalogue sources (scanned_clean only,
    # fail-closed) so the loop can search data MIXES, not just one fixed file. Clean-by-construction,
    # so it takes precedence over (and skips) the manifest-clean path below.
    if do_train and runtime.get("materialize_mixture"):
        mixed, mix_err = _materialize_data_mixture(config, out_dir, cmd_log)
        if mix_err:
            return fail(f"data-mixture: {mix_err}")
        if mixed:
            train_pairs = mixed
    else:
        # Leakage-safety: when a verified-clean prepared manifest is supplied, REAL training must use
        # the file the leakage gate certified — not the raw pairs the base config points at. This ties
        # "leakage: clean" to the data actually trained on (fail-closed if the cleaned file is missing).
        clean_pairs, clean_err = _clean_train_pairs_from_manifest(runtime)
        if clean_err:
            return fail(f"leakage-safety: {clean_err}")
        if do_train and clean_pairs:
            cmd_log.append({"note": "training on manifest-certified clean data",
                            "train_pairs_override": clean_pairs,
                            "train_pairs_configured": train_pairs})
            train_pairs = clean_pairs
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
        train_cmd = _build_train_cmd(train_script, train_base=train_base, train_pairs=train_pairs,
                                     hard_negs=hard_negs, ckpt=ckpt, steps=steps, plan=plan,
                                     run_id=run_id)
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
    # A trial is "ok" if it produced the metrics for the eval sets it was actually asked to run.
    # WebFAQ is only required when it was requested (a germanquad/dt_test-only config legitimately
    # has no WebFAQ metric and must not be marked failed).
    webfaq_requested = "webfaq" in eval_sets
    trial_ok = (not webfaq_requested) or (metrics["webfaq"].get("recall@100") is not None)
    result = {
        "run_id": run_id,
        "status": "ok" if trial_ok else "fail",
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
