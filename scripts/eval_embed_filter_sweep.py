#!/usr/bin/env python3
"""v7 EmbedFilter sweep — head-to-head: full 1024-d vs prefix Matryoshka vs EmbedFilter, at equal
dims, on the candidate dense embedder. Encodes each eval set's corpus+queries ONCE (un-normalized),
then ranks every method cheaply by matmul.

Stdlib core (config + planning + delta math); ``torch``/``sentence_transformers`` are imported only
inside the real run. ``--dry-run`` parses the config, checks artifact paths, and imports no ML.
Reuses ``eval_v6_1_dense_top50`` (``EVAL_SETS``/``_read``/``_qrels``/``eval_rankings``) so eval-set
paths and metric definitions match the rest of the repo. Never auto-promotes — advisory only.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import embed_filter as EF  # noqa: E402
from boldt_embed import experiment_registry as registry  # noqa: E402

# config eval-set name -> eval_v6_1_dense_top50.EVAL_SETS key
NAME_MAP = {"webfaq_heldout": "webfaq", "webfaq": "webfaq", "germanquad": "germanquad",
            "dt_test": "dt_test", "gerdalir": "gerdalir", "local_rag": "local_rag"}
# Eval sets not in the protected eval script's EVAL_SETS (e.g. the diagnostic-only GerDaLIR legal
# corpus) get their paths here; (corpus, queries, qrels, role). ev.EVAL_SETS takes precedence.
EXTRA_EVAL_SETS = {
    "gerdalir": ("data/processed/eval/gerdalir_corpus.jsonl",
                 "data/processed/eval/gerdalir_queries.jsonl",
                 "data/processed/eval/gerdalir_qrels.jsonl", "diagnostic"),
}
DEFAULT_BASIS_TEMPLATE = "outputs/embedfilter/boldt-dc-350m_tau{tau}"


def _load_eval_module():
    """Load the eval script: EVAL_SETS/_read/_qrels/eval_rankings (no torch at import)."""
    spec = importlib.util.spec_from_file_location(
        "eval_v6_1_dense_top50", ROOT / "scripts" / "eval_v6_1_dense_top50.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_sweep_config(path: str) -> Dict[str, Any]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def resolve_eval_sets(names: List[str], ev_eval_sets: Dict[str, Any]
                      ) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Split requested eval-set names into runnable (files present) vs skipped (with a reason)."""
    runnable, skipped = [], []
    for name in names:
        key = NAME_MAP.get(name)
        spec = ev_eval_sets.get(key) if key else None
        if not spec:
            skipped.append({"name": name, "reason": "no eval set defined on disk"})
            continue
        corpus_p, queries_p, qrels_p = spec[0], spec[1], spec[2]
        missing = [p for p in (corpus_p, queries_p, qrels_p) if not (ROOT / p).exists()]
        if missing:
            skipped.append({"name": name, "reason": f"missing files: {missing}"})
            continue
        runnable.append({"name": name, "key": key, "corpus": corpus_p,
                         "queries": queries_p, "qrels": qrels_p, "role": spec[3]})
    return runnable, skipped


def _basis_paths(taus: List[int], template: str) -> Dict[int, str]:
    return {t: template.format(tau=t) for t in taus}


def plan(config: Dict[str, Any], basis_template: str) -> Dict[str, Any]:
    """Dry-run planner: resolve eval sets + basis artifacts + methods, import no ML."""
    ev = _load_eval_module()
    active = list(config.get("active_eval_sets", []))
    diagnostic = list(config.get("diagnostic_eval_sets", []))
    all_sets = {**EXTRA_EVAL_SETS, **dict(ev.EVAL_SETS)}
    runnable, skipped = resolve_eval_sets(active + diagnostic, all_sets)
    taus = list(config.get("taus", []))
    bases = _basis_paths(taus, basis_template)
    basis_status = {t: {"path": p, "exists": (ROOT / p / "basis.pt").exists()}
                    for t, p in bases.items()}
    cand = config.get("candidate_embedder", "")
    return {
        "status": "dry_run",
        "candidate_embedder": cand,
        "candidate_present": bool(cand) and (ROOT / cand).exists(),
        "methods": ["full", "prefix" + str(config.get("compare_prefix_dims")),
                    "embedfilter(tau=" + ",".join(map(str, taus)) + ")"],
        "eval_sets_runnable": [r["name"] for r in runnable],
        "eval_sets_skipped": skipped,
        "basis_artifacts": basis_status,
        "promotion_policy": config.get("promotion_policy"),
    }


def compute_deltas(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Annotate each row with Δ vs full and (for embedfilter) Δ vs prefix at the same dim."""
    by_set: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        by_set.setdefault(r["eval_set"], {})
        if r["method"] == "full":
            by_set[r["eval_set"]]["full"] = r
        if r["method"] == "prefix":
            by_set[r["eval_set"]][f"prefix@{r['dim']}"] = r
    for r in rows:
        ref = by_set.get(r["eval_set"], {})
        full = ref.get("full")
        if full:
            r["dNDCG10_vs_full"] = round(r["ndcg@10"] - full["ndcg@10"], 4)
            r["dRecall100_vs_full"] = round(r["recall@100"] - full["recall@100"], 4)
        if r["method"] == "embedfilter":
            pref = ref.get(f"prefix@{r['dim']}")
            if pref:
                r["dNDCG10_vs_prefix"] = round(r["ndcg@10"] - pref["ndcg@10"], 4)
                r["dRecall100_vs_prefix"] = round(r["recall@100"] - pref["recall@100"], 4)
    return rows


# ------------------------------------------------------------------------------- real run (ML)
def _encode_raw(model, texts, batch_size=256):
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=False,
                        convert_to_tensor=True, show_progress_bar=False)


def _metrics_for(ev, q_emb, c_emb, cids, queries, qrels):
    import torch
    maxk = max(ev.KS)
    rankings = {}
    for start in range(0, q_emb.size(0), 256):
        sims = q_emb[start:start + 256] @ c_emb.t()
        idx = torch.topk(sims, min(maxk, len(cids)), dim=1).indices.tolist()
        for j, cand in enumerate(idx):
            rankings[str(queries[start + j]["query_id"])] = [cids[k] for k in cand]
    return ev.eval_rankings(rankings, qrels)


def run_sweep(config: Dict[str, Any], basis_template: str, limit: Optional[int]) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer

    ev = _load_eval_module()
    active = list(config.get("active_eval_sets", []))
    diagnostic = list(config.get("diagnostic_eval_sets", []))
    all_sets = {**EXTRA_EVAL_SETS, **dict(ev.EVAL_SETS)}
    runnable, skipped = resolve_eval_sets(active + diagnostic, all_sets)
    diag_names = set(diagnostic)

    cand = config["candidate_embedder"]
    cand_path = str((ROOT / cand)) if not pathlib.Path(cand).is_absolute() else cand
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(cand_path, device=dev)
    model.max_seq_length = 256

    prefix_dims = [d for d in config.get("compare_prefix_dims", [])]
    taus = list(config.get("taus", []))
    bases = {}
    for t, p in _basis_paths(taus, basis_template).items():
        bp = ROOT / p
        if (bp / "basis.pt").exists():
            basis, _meta = EF.load_embed_filter_basis(str(bp))
            bases[t] = basis.to(dev, torch.float32)

    rows: List[Dict[str, Any]] = []
    for rs in runnable:
        corpus = ev._read(str(ROOT / rs["corpus"]))
        queries = ev._read(str(ROOT / rs["queries"]))
        qrels = ev._qrels(str(ROOT / rs["qrels"]), queries)
        if limit:
            queries = queries[:limit]
            qids = {str(q["query_id"]) for q in queries}
            qrels = {k: v for k, v in qrels.items() if k in qids}
        cids = [c["doc_id"] for c in corpus]
        c_raw = _encode_raw(model, [c.get("text", "") for c in corpus]).to(torch.float32)
        q_raw = _encode_raw(model, [q.get("query", "") for q in queries]).to(torch.float32)
        H = c_raw.shape[1]

        def _row(method, dim, tau, m):
            return {"method": method, "dim": int(dim), "tau": tau, "eval_set": rs["name"],
                    "role": "diagnostic" if rs["name"] in diag_names else "active",
                    "ndcg@10": m["ndcg@10"], "recall@100": m["recall@100"], "mrr@10": m["mrr@10"],
                    "bytes_per_vector": int(dim) * 4}

        # full
        qf, cf = F.normalize(q_raw, dim=1), F.normalize(c_raw, dim=1)
        rows.append(_row("full", H, None, _metrics_for(ev, qf, cf, cids, queries, qrels)))
        # prefix Matryoshka (skip d>=H — d==H is 'full')
        for d in prefix_dims:
            if d >= H:
                continue
            m = _metrics_for(ev, F.normalize(q_raw[:, :d], dim=1), F.normalize(c_raw[:, :d], dim=1),
                             cids, queries, qrels)
            rows.append(_row("prefix", d, None, m))
        # EmbedFilter
        for t, basis in bases.items():
            qe, ce = F.normalize(q_raw @ basis, dim=1), F.normalize(c_raw @ basis, dim=1)
            m = _metrics_for(ev, qe, ce, cids, queries, qrels)
            rows.append(_row("embedfilter", basis.shape[1], t, m))
        print(f"[v7-sweep] {rs['name']} ({rs['role']}): H={H} q={len(queries)} c={len(cids)} done")

    rows = compute_deltas(rows)
    return {"status": "ok", "candidate_embedder": cand, "device": dev,
            "eval_sets_run": [r["name"] for r in runnable], "skipped": skipped,
            "rows": rows, "config": config}


def to_markdown(result: Dict[str, Any]) -> str:
    lines = ["# v7 EmbedFilter sweep", "",
             f"_Candidate embedder: `{result.get('candidate_embedder')}`. "
             "EmbedFilter basis = SVD bulk slice of the base unembedding, applied as a "
             "postprocessor. Numbers are real only if produced by a saved run._", ""]
    if result.get("skipped"):
        lines.append("Skipped eval sets: " + ", ".join(
            f"{s['name']} ({s['reason']})" for s in result["skipped"]) + "\n")
    lines += ["| eval set | role | method | dim | tau | nDCG@10 | Recall@100 | Δndcg/full | "
              "Δndcg/prefix | bytes/vec |", "|---|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for r in result.get("rows", []):
        lines.append(
            f"| {r['eval_set']} | {r['role']} | {r['method']} | {r['dim']} | {r.get('tau') or ''} "
            f"| {r['ndcg@10']} | {r['recall@100']} | {r.get('dNDCG10_vs_full', '')} "
            f"| {r.get('dNDCG10_vs_prefix', '')} | {r['bytes_per_vector']} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/experiments/v7_embedfilter.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs/v7-embedfilter/sweep.json"))
    ap.add_argument("--basis-template", default=DEFAULT_BASIS_TEMPLATE)
    ap.add_argument("--limit", type=int, default=None, help="cap queries per eval set (quick runs)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    config = load_sweep_config(args.config)
    if args.dry_run:
        print(json.dumps(plan(config, args.basis_template), ensure_ascii=False, indent=2))
        return 0

    result = run_sweep(config, args.basis_template, args.limit)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(to_markdown(result), encoding="utf-8")
    try:
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
        registry.emit_run_card(
            run_id=f"v7-embedfilter-sweep-{stamp}",
            run_type="eval", command="python " + " ".join(sys.argv),
            model=result["candidate_embedder"],
            dataset=",".join(result["eval_sets_run"]),
            metrics={"rows": len(result["rows"])},
            input_artifacts=[args.config], output_artifacts=[str(out)],
            notes="v7 EmbedFilter head-to-head sweep (full vs prefix vs embedfilter)")
    except Exception:
        pass
    print(f"[v7-sweep] {len(result['rows'])} rows over {result['eval_sets_run']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
