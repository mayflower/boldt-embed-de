#!/usr/bin/env python3
"""Mine v6.1 dense-specific hard negatives for WebFAQ Recall@50 (DENSE-ONLY — no reranker training).

Retrieves dense-Boldt-v6 rankings over the corpus, finds queries whose positive sits at dense rank
51..window (in top-100/200 but not top-50), and mines the docs that outrank it (+ BM25 confusions +
teacher-confirmed hard negatives), with a false-negative veto from existing Qwen3-Reranker-8B teacher
scores. `--dry-run` imports NO ML and mines from precomputed `dense_ranked` lists in the input.

Writes data/processed/v6_1/dense_top50_hardnegatives.jsonl + a report.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import dense_top50_mining as M  # noqa: E402
from boldt_embed.v6_1_dense_config import load_v6_1_dense_config  # noqa: E402


def _read(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]


def _load_qrels(p, queries):
    pos = {}
    if p:
        for r in _read(p):
            try:
                rel = float(r.get("relevance", 1))
            except (TypeError, ValueError):
                rel = 1.0
            if rel > 0 and r.get("doc_id"):
                pos.setdefault(str(r["query_id"]), set()).add(r["doc_id"])
    for q in queries:
        ids = q.get("positive_doc_ids") or ([q["positive_doc_id"]] if q.get("positive_doc_id") else [])
        if ids:
            pos.setdefault(str(q["query_id"]), set()).update(ids)
    return pos


def _load_teacher_scores(p):
    """Nested {query_id: {doc_id: teacher_score}} from a teacher-scored candidate-list file."""
    ts = {}
    if not p or not pathlib.Path(p).exists():
        return ts
    for r in _read(p):
        qid = str(r.get("query_id"))
        m = ts.setdefault(qid, {})
        for c in r.get("candidates", []):
            if c.get("teacher_score") is not None and c.get("doc_id"):
                m[c["doc_id"]] = float(c["teacher_score"])
    return ts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiments" / "v6_1_dense_top50.json"))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qrels", default=None)
    ap.add_argument("--teacher-scores", default="data/processed/v6/reranker_train_lists_teacher_scored.jsonl",
                    help="existing Qwen3-Reranker-8B teacher scores (for veto + teacher hard negs)")
    ap.add_argument("--dense-model", default="outputs/v6-dense-rag/checkpoints/boldt-dense-rag-v6")
    ap.add_argument("--with-bm25", action="store_true", help="also mine BM25 lexical confusions")
    ap.add_argument("--domain", default="faq_real")
    ap.add_argument("--output", default="data/processed/v6_1/dense_top50_hardnegatives.jsonl")
    ap.add_argument("--report", default="outputs/v6-dense-rag/v6_1_top50_hardneg_report.json")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--top50", type=int, default=50)
    ap.add_argument("--veto-margin", type=float, default=2.0)
    ap.add_argument("--max-negatives", type=int, default=20)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true",
                    help="mine from precomputed dense_ranked in --queries; imports NO ML")
    args = ap.parse_args()

    cfg = load_v6_1_dense_config(args.config)         # fail-closed; reranker_training_enabled=false
    assert cfg["reranker_training_enabled"] is False, "v6.1 is dense-only"

    corpus_rows = _read(args.corpus)
    corpus = {c["doc_id"]: c.get("text", "") for c in corpus_rows}
    queries = _read(args.queries)
    qrels = _load_qrels(args.qrels, queries)
    teacher = _load_teacher_scores(args.teacher_scores)

    qrecs = [{"query_id": str(q["query_id"]), "query": q.get("query", ""),
              "positive_doc_id": (sorted(qrels.get(str(q["query_id"]), {None}))[0]),
              "domain": q.get("domain", args.domain), "source": "webfaq_train"} for q in queries]

    if args.dry_run:
        # use precomputed dense_ranked (+bm25_ranked) supplied on the query rows; no ML
        by_id = {str(q["query_id"]): q for q in queries}
        for r in qrecs:
            q = by_id[r["query_id"]]
            r["dense_ranked"] = q.get("dense_ranked") or []
            if q.get("bm25_ranked"):
                r["bm25_ranked"] = q["bm25_ranked"]
        assert "torch" not in sys.modules, "dry-run must not import torch"
    else:
        import torch
        from sentence_transformers import SentenceTransformer
        from boldt_embed import bm25_index as BM
        m = SentenceTransformer(args.dense_model, device="cuda" if torch.cuda.is_available() else "cpu")
        m.max_seq_length = args.max_seq_length
        dids = [c["doc_id"] for c in corpus_rows]
        demb = m.encode([corpus[d] for d in dids], batch_size=256, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)
        qemb = m.encode([q.get("query", "") for q in queries], batch_size=256,
                        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        dt = torch.tensor(demb)
        bm = BM.build_bm25_index(corpus_rows, text_field="text", id_field="doc_id",
                                 fold_umlauts=True) if args.with_bm25 else None
        for i, r in enumerate(qrecs):
            sims = torch.tensor(qemb[i]) @ dt.T
            top = torch.topk(sims, k=min(args.window, len(dids))).indices.tolist()
            r["dense_ranked"] = [dids[j] for j in top]
            if bm is not None:
                r["bm25_ranked"] = [d for d, _ in bm.search(queries[i].get("query", ""), top_k=50)]

    out = M.mine_set(qrecs, corpus, qrels=qrels, teacher_scores=teacher, top50=args.top50,
                     window=args.window, veto_margin=args.veto_margin,
                     max_negatives=args.max_negatives)
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in out["records"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pathlib.Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.report).write_text(json.dumps(out["report"], ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    rep = out["report"]
    print(f"[v6.1-top50-mine] mined {rep['queries_mined']} target queries "
          f"(rank51-100={rep['positive_rank_51_100']}, rank101-200={rep['positive_rank_101_200']}) "
          f"-> {args.output}")
    print(f"[v6.1-top50-mine] negatives={rep['total_negatives']} "
          f"(avg {rep['avg_negatives_per_query']}/q) by_source={rep['negatives_by_source']} "
          f"veto={rep['false_negative_veto_count']} leakage_excluded={rep['leakage_excluded']}")
    print(f"[v6.1-top50-mine] teacher_margin_dist={rep['teacher_margin_distribution']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
