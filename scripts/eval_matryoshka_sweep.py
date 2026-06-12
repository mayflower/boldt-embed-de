#!/usr/bin/env python3
"""Matryoshka dimension sweep for one embedder on one retrieval set.

Encodes corpus+queries ONCE at full dim, then for each Matryoshka dim truncates +
renormalizes (eval_harness.truncate_normalized) and ranks via GPU matmul, reporting nDCG@10
and recall. Writes the v1-compatible {"matryoshka_sweep": {dim: {...}}} schema so
summarize_v2_results / the release gate can read it. Needs the `eval` extras + a GPU.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import eval_harness as eh  # noqa: E402
from boldt_embed.experiment_registry import emit_run_card  # noqa: E402


def _read(path):
    return [json.loads(l) for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval-corpus", required=True)
    ap.add_argument("--eval-queries", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--dataset", default="germanquad")
    ap.add_argument("--dims", default="1024,768,512,256,128,64")
    ap.add_argument("--query-instruction", default="")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--output", required=True)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    corpus = _read(args.eval_corpus)
    queries = _read(args.eval_queries)
    qrels = _read(args.qrels)
    pos = {}
    for r in qrels:
        if float(r.get("relevance", 1)) > 0:
            pos.setdefault(str(r["query_id"]), set()).add(str(r["doc_id"]))

    cids = [str(c.get("doc_id") or c.get("id")) for c in corpus]
    d_texts = [c.get("text") or c.get("document") or "" for c in corpus]
    q_texts = [args.query_instruction + q["query"] for q in queries]
    qids = [str(q["query_id"]) for q in queries]

    model = SentenceTransformer(args.model, device="cuda")
    model.max_seq_length = args.max_length
    print(f"[encode] {len(d_texts)} docs + {len(q_texts)} queries with {args.model}")
    c_full = model.encode(d_texts, batch_size=args.batch_size, normalize_embeddings=True,
                          show_progress_bar=False, convert_to_numpy=True)
    q_full = model.encode(q_texts, batch_size=args.batch_size, normalize_embeddings=True,
                          show_progress_bar=False, convert_to_numpy=True)

    dims = [int(x) for x in args.dims.split(",")]
    sweep = {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for dim in dims:
        # truncate + renormalize (Matryoshka prefix), then GPU matmul rank
        c_t = torch.tensor([eh.truncate_normalized(v, dim) for v in c_full], dtype=torch.float32, device=dev)
        q_t = torch.tensor([eh.truncate_normalized(v, dim) for v in q_full], dtype=torch.float32, device=dev)
        topk = min(200, len(cids))
        rows = []
        for s in range(0, q_t.size(0), 256):
            sims = q_t[s:s + 256] @ c_t.t()
            idx = torch.topk(sims, topk, dim=1).indices.tolist()
            for j, row in enumerate(idx):
                ranked = [cids[k] for k in row]
                rows.append(eh.metrics_for_query(ranked, pos.get(qids[s + j], set()), (1, 3, 5, 10, 100)))
        agg = eh.aggregate(rows)
        sweep[str(dim)] = {"ndcg@10": round(agg.get("ndcg@10", 0.0), 4),
                           "recall@10": round(agg.get("recall@10", 0.0), 4),
                           "recall@100": round(agg.get("recall@100", 0.0), 4)}
        print(f"  dim {dim:>4}: ndcg@10={sweep[str(dim)]['ndcg@10']}")

    base = sweep.get(str(dims[0]), {}).get("ndcg@10") or 0.0
    out = {"status": "ok", "model": args.model, "dataset": args.dataset,
           "n_queries": len(queries), "n_corpus": len(corpus), "matryoshka_sweep": sweep,
           "retention_vs_full": {d: round((v["ndcg@10"] / base) if base else 0.0, 4)
                                 for d, v in sweep.items()}}
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {args.output}")
    if args.run_id:
        emit_run_card(args.run_id, "eval", "scripts/eval_matryoshka_sweep.py",
                      model=args.model, dataset=args.dataset,
                      metrics={f"ndcg@10_dim{d}": v["ndcg@10"] for d, v in sweep.items()},
                      output_artifacts=[args.output])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
