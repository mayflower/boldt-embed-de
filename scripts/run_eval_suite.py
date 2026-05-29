#!/usr/bin/env python3
"""German evaluation suite: STS, classification, clustering, cross-lingual, RAG, stress,
and Matryoshka efficiency (prompt 09).

Default encoder is the deterministic HashingEncoder STAND-IN (stdlib, NOT Boldt) so the
suite runs in CI without weights. Pass --model <checkpoint> to evaluate the REAL trained
model (needs the train extras + GPU). Public MMTEB stays a separate scaffold (run_mteb_*).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import data as datamod  # noqa: E402
from boldt_embed import eval_harness as eh  # noqa: E402

B = ROOT / "benchmarks"
EMBED_DIM = 256
MATRYOSHKA = (256, 128, 64)


def build_encoder(model_path, device_index, pooling):
    if not model_path:
        enc = eh.HashingEncoder(dim=EMBED_DIM, ngram=3)
        return enc.encode, f"HashingEncoder(dim={EMBED_DIM}) STAND-IN (not Boldt)", EMBED_DIM
    from boldt_embed import train as T

    def encode(texts):
        return T.encode_texts(model_path, texts, pooling=pooling, device_index=device_index)

    return encode, f"real model: {model_path} (pooling={pooling})", 1024


def run(model_path=None, device_index=0, pooling="eos") -> dict:
    encode, encoder_name, dim = build_encoder(model_path, device_index, pooling)

    cls = datamod.load_jsonl(B / "classification_de.jsonl")
    return {
        "encoder": encoder_name,
        "disclaimer": "Default HashingEncoder validates plumbing only; not a Boldt quality claim.",
        "sts": eh.evaluate_sts(datamod.load_jsonl(B / "sts_de.jsonl"), encode),
        "classification": eh.evaluate_classification(
            [r for r in cls if r["split"] == "train"],
            [r for r in cls if r["split"] == "test"], encode),
        "clustering": eh.evaluate_clustering(datamod.load_jsonl(B / "clustering_de.jsonl"), encode, k=3),
        "crosslingual_de_en": eh.retrieval_with_encoder(
            json.loads((B / "crosslingual_deen.json").read_text("utf-8")), encode),
        "rag": eh.retrieval_with_encoder(json.loads((B / "rag_de.json").read_text("utf-8")), encode),
        "stress_bm25": eh.evaluate_stress(json.loads((B / "stress_de.json").read_text("utf-8"))),
        "efficiency_matryoshka": eh.efficiency_report(dim, MATRYOSHKA if dim == 256 else (1024, 512, 256, 128, 64)),
    }


def render_markdown(r: dict) -> str:
    L = ["# German Evaluation Suite", "", f"Encoder: `{r['encoder']}`", "", f"> {r['disclaimer']}", "",
         "| Task | Metric | Value |", "|---|---|---:|",
         f"| STS | spearman | {r['sts']['spearman']:.4f} |",
         f"| Classification | accuracy | {r['classification']['accuracy']:.4f} |",
         f"| Clustering | v_measure | {r['clustering']['v_measure']:.4f} |",
         f"| Cross-lingual DE→EN | ndcg@10 | {r['crosslingual_de_en']['ndcg@10']:.4f} |",
         f"| RAG | ndcg@10 | {r['rag']['ndcg@10']:.4f} |",
         f"| Stress (BM25) | ndcg@10 | {r['stress_bm25']['overall']['ndcg@10']:.4f} |", "",
         "## Stress by case (BM25)", "", "| Case | recall@1 | ndcg@10 |", "|---|---:|---:|"]
    for case, m in r["stress_bm25"]["by_case"].items():
        L.append(f"| {case} | {m['recall@1']:.2f} | {m['ndcg@10']:.4f} |")
    return "\n".join(L)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="trained checkpoint dir (real eval)")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--pooling", default="eos")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    report = run(args.model, args.device_index, args.pooling)
    if args.save:
        out = ROOT / "outputs" / "benchmarks"
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval-suite-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
        (out / "eval-suite-report.md").write_text(render_markdown(report), "utf-8")
    print(render_markdown(report) if args.format == "markdown" else json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
